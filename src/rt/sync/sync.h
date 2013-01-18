// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
