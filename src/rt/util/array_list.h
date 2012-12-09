// -*- c++ -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef ARRAY_LIST_H
#define ARRAY_LIST_H

#include <inttypes.h>
#include <stddef.h>

/**
 * A simple, resizable array list. Note that this only works with POD types
 * (because data is grown via realloc).
 */
template<typename T> class array_list {
    static const size_t INITIAL_CAPACITY = 8;
    size_t _size;
    T * _data;
    size_t _capacity;
private:
    // private and left undefined to disable copying
    array_list(const array_list& rhs);
    array_list& operator=(const array_list& rhs);
public:
    array_list();
    ~array_list();
    size_t size() const;
    int32_t append(T value);
    int32_t push(T value);
    bool pop(T *value);
    bool replace(T old_value, T new_value);
    int32_t index_of(T value) const;
    bool is_empty() const;
    T* data();
    const T* data() const;
    T & operator[](size_t index);
    const T & operator[](size_t index) const;
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
array_list<T>::size() const {
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
array_list<T>::index_of(T value) const {
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

template<typename T> const T &
array_list<T>::operator[](size_t index) const {
    return _data[index];
}

template<typename T> bool
array_list<T>::is_empty() const {
    return _size == 0;
}

template<typename T> T*
array_list<T>::data() {
    return _data;
}

template<typename T> const T*
array_list<T>::data() const {
    return _data;
}

#endif /* ARRAY_LIST_H */
