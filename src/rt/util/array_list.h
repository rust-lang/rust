// -*- c++ -*-
#ifndef ARRAY_LIST_H
#define ARRAY_LIST_H

/**
 * A simple, resizable array list.
 */
template<typename T> class array_list {
    static const size_t INITIAL_CAPACITY = 8;
    size_t _size;
    T * _data;
    size_t _capacity;
public:
    array_list();
    ~array_list();
    size_t size();
    int32_t append(T value);
    int32_t push(T value);
    bool pop(T *value);
    bool replace(T old_value, T new_value);
    int32_t index_of(T value);
    bool is_empty();
    T* data();
    T & operator[](size_t index);
};

template<typename T>
array_list<T>::array_list() {
    _size = 0;
    _capacity = INITIAL_CAPACITY;
    _data = (T *) malloc(sizeof(T) * _capacity);
}

template<typename T>
array_list<T>::~array_list() {
    free(_data);
}

template<typename T> size_t
array_list<T>::size() {
    return _size;
}

template<typename T> int32_t
array_list<T>::append(T value) {
    return push(value);
}

template<typename T> int32_t
array_list<T>::push(T value) {
    if (_size == _capacity) {
        _capacity = _capacity * 2;
        _data = (T *) realloc(_data, _capacity * sizeof(T));
    }
    _data[_size ++] = value;
    return _size - 1;
}

template<typename T> bool
array_list<T>::pop(T *value) {
    if (_size == 0) {
        return false;
    }
    if (value != NULL) {
        *value = _data[-- _size];
    } else {
        -- _size;
    }
    return true;
}

/**
 * Replaces the old_value in the list with the new_value.
 * Returns the true if the replacement succeeded, or false otherwise.
 */
template<typename T> bool
array_list<T>::replace(T old_value, T new_value) {
    int index = index_of(old_value);
    if (index < 0) {
        return false;
    }
    _data[index] = new_value;
    return true;
}

template<typename T> int32_t
array_list<T>::index_of(T value) {
    for (size_t i = 0; i < _size; i++) {
        if (_data[i] == value) {
            return i;
        }
    }
    return -1;
}

template<typename T> T &
array_list<T>::operator[](size_t index) {
    return _data[index];
}

template<typename T> bool
array_list<T>::is_empty() {
    return _size == 0;
}

template<typename T> T*
array_list<T>::data() {
    return _data;
}

#endif /* ARRAY_LIST_H */
