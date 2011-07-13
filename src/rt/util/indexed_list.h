// -*- c++ -*-
#ifndef INDEXED_LIST_H
#define INDEXED_LIST_H

#include <assert.h>
#include "array_list.h"

class indexed_list_object {
public:
    int32_t list_index;
};

template<typename T>
class indexed_list_element : public indexed_list_object {
public:
    T value;
    indexed_list_element(T value) : value(value) {
    }
};

/**
 * An array list of objects that are aware of their position in the list.
 * Normally, objects in this list should derive from the base class
 * "indexed_list_object" however because of nasty Rust compiler dependencies
 * on the layout of runtime objects we cannot always derive from this
 * base class, so instead we just enforce the informal protocol that any
 * object inserted in this list must define a "int32_t list_index" member.
 */
template<typename T> class indexed_list {
    array_list<T*> list;
public:
    virtual int32_t append(T *value);
    virtual bool pop(T **value);
    /**
     * Same as pop(), except that it returns NULL if the list is empty.
     */
    virtual T* pop_value();
    virtual size_t length() {
        return list.size();
    }
    virtual bool is_empty() {
        return list.is_empty();
    }
    virtual int32_t remove(T* value);
    virtual T * operator[](int32_t index);
};

template<typename T> int32_t
indexed_list<T>::append(T *value) {
    value->list_index = list.push(value);
    return value->list_index;
}

/**
 * Swap delete the last object in the list with the specified object.
 */
template<typename T> int32_t
indexed_list<T>::remove(T *value) {
    assert (value->list_index >= 0);
    assert (value->list_index < (int32_t)list.size());
    int32_t removeIndex = value->list_index;
    T *last = 0;
    list.pop(&last);
    if (last->list_index == removeIndex) {
        last->list_index = -1;
        return removeIndex;
    } else {
        value->list_index = -1;
        list[removeIndex] = last;
        last->list_index = removeIndex;
        return removeIndex;
    }
}

template<typename T> bool
indexed_list<T>::pop(T **value) {
    return list.pop(value);
}

template<typename T> T*
indexed_list<T>::pop_value() {
    T *value = NULL;
    if (list.pop(&value)) {
        return value;
    }
    return NULL;
}

template <typename T> T *
indexed_list<T>::operator[](int32_t index) {
    T *value = list[index];
    assert(value->list_index == index);
    return value;
}

#endif /* INDEXED_LIST_H */
