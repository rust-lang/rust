// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a `&mut` inside of an `&` is freezable.

struct MutSlice<'a, T:'a> {
    data: &'a mut [T]
}

fn get<'a, T>(ms: &'a MutSlice<'a, T>, index: uint) -> &'a T {
    &ms.data[index]
}

pub fn main() {
    let mut data = [1i, 2, 3];
    {
        let slice = MutSlice { data: data };
        slice.data[0] += 4;
        let index0 = get(&slice, 0);
        let index1 = get(&slice, 1);
        let index2 = get(&slice, 2);
        assert_eq!(*index0, 5);
        assert_eq!(*index1, 2);
        assert_eq!(*index2, 3);
    }
    assert_eq!(data[0], 5);
    assert_eq!(data[1], 2);
    assert_eq!(data[2], 3);
}
