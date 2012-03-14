// -*- c++ -*-
#ifndef SYNC_H
#define SYNC_H

class sync {
public:
    template <class T>
    static bool compare_and_swap(T *address,
        T oldValue, T newValue) {
        return __sync_bool_compare_and_swap(address, oldValue, newValue);
    }

    template <class T>
    static T increment(T *address) {
        return __sync_add_and_fetch(address, 1);
    }

    template <class T>
    static T decrement(T *address) {
        return __sync_sub_and_fetch(address, 1);
    }

    template <class T>
    static T increment(T &address) {
        return __sync_add_and_fetch(&address, 1);
    }

    template <class T>
    static T decrement(T &address) {
        return __sync_sub_and_fetch(&address, 1);
    }

    template <class T>
    static T read(T *address) {
        return __sync_add_and_fetch(address, 0);
    }

    template <class T>
    static T read(T &address) {
        return __sync_add_and_fetch(&address, 0);
    }
};

#endif /* SYNC_H */
