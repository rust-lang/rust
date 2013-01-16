// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

fn vec_peek<T>(v: &r/[T]) -> Option< (&r/T, &r/[T]) > {
    if v.len() == 0 {
        None
    } else {
        let head = &v[0];
        let tail = v.view(1, v.len());
        Some( (head, tail) )
    }
}


fn test_peek_empty_stack() {
    let v : &[int] = &[];
    assert (None == vec_peek(v));
}

fn test_peek_empty_unique() {
    let v : ~[int] = ~[];
    assert (None == vec_peek(v));
}

fn test_peek_empty_managed() {
    let v : @[int] = @[];
    assert (None == vec_peek(v));
}


fn main() {}
