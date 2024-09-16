import meep as mp
import numpy as np
from matplotlib import pyplot as plt
from plotting import plot_eigenmode

dpml = 1

slab_wg_l = 2
slab_wg_w = .45
slab_wg_h = .22

rib_slab_h = .09
rib_l = 2
rib_w = .3
rib_h = slab_wg_h - rib_slab_h

si = mp.Medium(index=3.47)
siox = mp.Medium(index=1.44)

rib_waveguide = [
    mp.Block(
        center=mp.Vector3(0,0,-slab_wg_h/2 + rib_slab_h/2),
        size=mp.Vector3(rib_l, 3*slab_wg_w, rib_slab_h),
        material=si
    ),
    mp.Block(
        center=mp.Vector3(0,0,slab_wg_h/2 - rib_h/2),
        size=mp.Vector3(rib_l,rib_w,rib_h),
        material=si
    )
]

input_wg = [
    mp.Block(
        center=mp.Vector3(-rib_l/2-slab_wg_l/2),
        size=mp.Vector3(slab_wg_l,slab_wg_w,slab_wg_h),
        material=si
    )
]
output_wg = [
    mp.Block(
        center=mp.Vector3(rib_l/2+slab_wg_l/2),
        size=mp.Vector3(slab_wg_l,slab_wg_w,slab_wg_h),
        material=si
    )
]


cell_size = mp.Vector3(
    sx:=2*slab_wg_l + rib_l + 2*dpml,
    sy:=6*slab_wg_w+2*dpml,
    sz:=slab_wg_h+.5+2*dpml
)

wavcen = 1.55
fcen = 1/wavcen
fwidth = 0.2 * fcen

sources = [
    mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(-rib_l/2 - slab_wg_l/2),
        size=mp.Vector3(0,5*slab_wg_w,3*slab_wg_h),
        eig_band=1,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=64,
    geometry=rib_waveguide + input_wg + output_wg,
    sources=sources,
    default_material=siox,
    boundary_layers=[mp.PML(dpml)]
)

in_mon = sim.add_mode_monitor(
    freqs:=np.linspace(fcen-fwidth/2, fcen+fwidth/2, 21),
    mp.ModeRegion(
        center=mp.Vector3(-rib_l/2 -slab_wg_l/2 +.5),
        size=mp.Vector3(0,5*slab_wg_w,3*slab_wg_h),
        # weight=-1
    )
)

out_mon = sim.add_mode_monitor(
    freqs:=np.linspace(fcen-fwidth/2, fcen+fwidth/2, 21),
    mp.ModeRegion(
        center=mp.Vector3(rib_l/2 +slab_wg_l/2+.5),
        size=mp.Vector3(0,5*slab_wg_w,3*slab_wg_h),
        # weight=-1
    )
)

# print(sim.cell_size)
# sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(sx,sy)))
# plt.show()
# quit()

sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-4))

np.savez('polarizer_data.npz', in_mon=sim.get_eigenmode_coefficients(in_mon,[1,2]).alpha, out_mon=sim.get_eigenmode_coefficients(out_mon,[1,2]).alpha, freqs=freqs)

# for band in [1,2]:
#     f = plot_eigenmode(
#         sim,
#         mp.Volume(
#             center=mp.Vector3(-rib_l/2 - slab_wg_l/2),
#             size=mp.Vector3(0,2*slab_wg_w,2*slab_wg_h)
#         ),
#         band=band,
#         three_d=True,
#         wavelength=wavcen
#     )

#     f.savefig(f'slab_mode_{band}.png',dpi=300)

#     f = plot_eigenmode(
#         sim,
#         mp.Volume(
#             center=mp.Vector3(),
#             size=mp.Vector3(0,2*slab_wg_w,2*slab_wg_h)
#         ),
#         band=band,
#         three_d=True,
#         wavelength=wavcen
#     )

#     f.savefig(f'ribg_mode_{band}.png',dpi=300)

