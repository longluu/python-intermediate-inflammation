import numpy as np

### Procedural programming
# Patient data
# data = np.array([[1., 2., 3.],
#                  [4., 5., 6.]])
#
#
# def attach_names(data, names):
#     """ Function to attach name to data """
#     patients = []
#     n_max_item = np.max([data.shape[0], len(names)])
#     for index in range(n_max_item):
#         if index > data.shape[0] - 1:
#             patients.append({'name': names[index],
#                              'data': np.array([])})
#         elif index > len(names) - 1:
#             patients.append({'name': '',
#                              'data': data[index]})
#         else:
#             patients.append({'name': names[index],
#                              'data': data[index]})
#     return patients
#
#
# output = attach_names(data, ['Alice', 'Long', 'Linh'])
# print(output)

class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return self.title + " by " + self.author

book = Book('Beyond Order', 'Jordan Peterson')
print(book)