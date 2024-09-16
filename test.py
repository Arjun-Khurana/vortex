import numpy as np
from matplotlib import pyplot as plt
import h5py

# farfields = np.load('farfields.npz')
# print(farfields.keys())
# print(np.shape(farfields['ex']))

# intensity_z = (
#     np.absolute(farfields["ex"]) ** 2
#     + np.absolute(farfields["ey"]) ** 2
#     + np.absolute(farfields["ez"]) ** 2
# )

# plt.imshow(intensity_z[...,7])
# plt.colorbar()
# # print(np.shape(intensity_z))
# plt.savefig('farfields-7.png',dpi=300)

data = np.load('polarizer_data_3.npz')
print(np.shape(data['in_mon']))
plt.plot(np.abs(np.divide(data['out_mon'][0,:,0], data['in_mon'][0,:,0])**2))

plt.plot(np.abs(np.divide(data['out_mon'][1,:,0], data['in_mon'][1,:,0])**2))
plt.show()