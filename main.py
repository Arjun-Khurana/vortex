import numpy as np
import meep as mp
from matplotlib import pyplot as plt
import argparse

def main(args):
    resolution = 30
    r = 3
    wg_width = .5
    wg_height = .22
    c_g = .05
    dpml = 1
    sx = 2*(r + wg_width + 1 + dpml)
    sy = 2*(r + wg_width + c_g + wg_width + 1 + dpml)
    sz = wg_height + 1 + 2*dpml
    si = mp.Medium(index=3.5)
    siox = mp.Medium(index=1.4)
    src_offset=0.5

    ring = [
        mp.Cylinder(radius=r+wg_width/2, height=wg_height, material=si),
        mp.Cylinder(radius=r-wg_width/2, material=siox)
    ]

    wg = [
        mp.Block(
            center=mp.Vector3(0,-r-wg_width-c_g), 
            size=mp.Vector3(sx, wg_width, wg_height),
            material=si
        )
    ]

    N = 30
    thetas = np.linspace(0,2*np.pi,N)[1:]
    ds = .3
    dr = .2
    dtheta = np.arcsin(ds/r)
    
    grating = [
        mp.Prism(
            vertices=[
                mp.Vector3(0,0,-wg_height/2),
                mp.Vector3(r*np.cos(theta - dtheta/2), r*np.sin(theta - dtheta/2),-wg_height/2),
                mp.Vector3(r*np.cos(theta + dtheta/2), r*np.sin(theta + dtheta/2),-wg_height/2),
            ],
            height=wg_height,
            material=si
        )
        for theta in thetas
    ] + [
        mp.Cylinder(radius=r-wg_width/2-dr,material=siox)
    ]

    wavcen = 1.55
    fcen = 1/wavcen
    fwidth = 0.2*fcen

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            center=mp.Vector3(x=-sx/2 + dpml + src_offset,y=-r-wg_width-c_g),
            size=mp.Vector3(0,2*wg_width, 2*wg_height),
            direction=mp.NO_DIRECTION,
            eig_kpoint=mp.Vector3(1,0,0)
        )
    ]

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx,sy,sz),
        resolution=resolution,
        geometry=ring + wg + grating,
        sources=sources,
        boundary_layers=[mp.PML(dpml)],
        default_material=siox
    )

    n2f_monitor = sim.add_near2far(
        fcen,
        fwidth,
        11,
        mp.Near2FarRegion(
            center = mp.Vector3(0,0,sz/2 -dpml),
            size = mp.Vector3(sx,sy,0),
            weight=1
        )
    )

    sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(sx,sy)))
    # if mp.am_really_master(): 
    plt.savefig('setup.png',dpi=300)
    plt.close()

    animate = mp.Animate2D(
        fields=mp.Hz,
        realtime=False,
        normalize=True,
        output_plane=mp.Volume(mp.Vector3(),size=mp.Vector3(sx,sy))
    )
    animate_up = mp.Animate2D(
        fields=mp.Hz,
        realtime=False,
        normalize=True,
        output_plane=mp.Volume(mp.Vector3(z=.5),size=mp.Vector3(sx,sy))
    )

    field_array_x = []
    field_array_y = []
    def save_fields(sim:mp.Simulation):
        eps = sim.get_array(mp.Dielectric, vol=mp.Volume(mp.Vector3(),sim.cell_size))
        ex = sim.get_array(mp.Ex, center=mp.Vector3(), size=sim.cell_size)
        ey = sim.get_array(mp.Ey, center=mp.Vector3(), size=sim.cell_size)
        # field_array_x.append(ex)
        # field_array_y.append(ey)
        np.savez(f'fields/fields-{sim.timestep()//(2*resolution)}.npz', ex=ex, ey=ey)

    def save_epsilon(sim):
        eps = sim.get_array(mp.Dielectric, vol=mp.Volume(mp.Vector3(),sim.cell_size))
        np.savez(f'eps.npz', eps=eps)

    sim.run(
        mp.at_every(1, animate, animate_up), 
        until=mp.stop_when_fields_decayed(
            10, mp.Hz, mp.Vector3(0,0,sz/2-dpml),1e-5
        )
    )
    animate.to_gif(fps=15,filename='animation.gif')
    animate_up.to_gif(fps=15,filename='animation_up.gif')

    farfields = sim.get_farfields(
        n2f_monitor,
        resolution//5,
        center=mp.Vector3(
            0,
            0,
            20
        ),
        size=mp.Vector3(2*sx, 2*sy, 0),
    )

    intensity_z = (
        np.absolute(farfields["Ex"]) ** 2
        + np.absolute(farfields["Ey"]) ** 2
        + np.absolute(farfields["Ez"]) ** 2
    )

    np.savez(f'farfields.npz', 
        ex=farfields['Ex'], 
        ey=farfields['Ey'],
        ez=farfields['Ez'],
        hx=farfields['Hx'],
        hy=farfields['Hy'],
        hz=farfields['Hz'],
    )

    # sim.run(
    #     mp.at_beginning(save_epsilon), 
    #     # mp.to_appended("hz", mp.at_every(1, mp.output_hfield_z)), 
    #     mp.at_every(2,animate, animate_up, save_fields),
    #     until=200
    # )

    # sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(sx,sy)))
    # plt.show()
    # sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(0,sy,sz)))
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)