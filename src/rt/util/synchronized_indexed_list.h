#ifndef SYNCHRONIZED_INDEXED_LIST_H
#define SYNCHRONIZED_INDEXED_LIST_H

#include "indexed_list.h"
#include "../sync/lock_and_signal.h"

template<typename T> class synchronized_indexed_list :
    public indexed_list<T> {
    lock_and_signal _lock;

public:
    synchronized_indexed_list() {
    }

    int32_t append(T *value) {
        int32_t index = 0;
        _lock.lock();
        index = indexed_list<T>::append(value);
        _lock.unlock();
        return index;
    }

    bool pop(T **value) {
        _lock.lock();
        bool result = indexed_list<T>::pop(value);
        _lock.unlock();
        return result;
    }

    size_t length() {
       size_t length = 0;
       _lock.lock();
       length = indexed_list<T>::length();
       _lock.unlock();
       return length;
    }

    bool is_empty() {
        bool empty = false;
        _lock.lock();
        empty = indexed_list<T>::is_empty();
        _lock.unlock();
        return empty;
    }

    int32_t remove(T* value) {
        size_t index = 0;
        _lock.lock();
        index = indexed_list<T>::remove(value);
        _lock.unlock();
        return index;
    }

    T *operator[](size_t index) {
        T *value = NULL;
        _lock.lock();
        value = indexed_list<T>::operator[](index);
        _lock.unlock();
        return value;
    }
};

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

#endif /* SYNCHRONIZED_INDEXED_LIST_H */
