// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Entry<A,B> {
    key: A,
    value: B
}

pub struct alist<A,B> {
    eq_fn: extern "Rust" fn(A,A) -> bool,
    data: @mut ~[Entry<A,B>]
}

pub fn alist_add<A:'static,B:'static>(lst: &alist<A,B>, k: A, v: B) {
    lst.data.push(Entry{key:k, value:v});
}

pub fn alist_get<A:Clone + 'static,
                 B:Clone + 'static>(
                 lst: &alist<A,B>,
                 k: A)
                 -> B {
    let eq_fn = lst.eq_fn;
    for entry in lst.data.iter() {
        if eq_fn(entry.key.clone(), k.clone()) {
            return entry.value.clone();
        }
    }
    fail!();
}

#[inline]
pub fn new_int_alist<B:'static>() -> alist<int, B> {
    fn eq_int(a: int, b: int) -> bool { a == b }
    return alist {eq_fn: eq_int, data: @mut ~[]};
}

#[inline]
pub fn new_int_alist_2<B:'static>() -> alist<int, B> {
    #[inline]
    fn eq_int(a: int, b: int) -> bool { a == b }
    return alist {eq_fn: eq_int, data: @mut ~[]};
}
