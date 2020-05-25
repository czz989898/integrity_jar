import os
def del_file(path_data):
    for i in os.listdir(path_data) :
        file_data = path_data + "/" + i
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            del_file(file_data)
if __name__ == "__main__":
    #todo same_test()
    #todo diff_test()
    del_file('./data')
