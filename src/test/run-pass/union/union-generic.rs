// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

union MaybeItem<T: Iterator> {
    elem: T::Item,
    none: (),
}

union U<A, B> {
    a: A,
    b: B,
}

unsafe fn union_transmute<A, B>(a: A) -> B {
    U { a: a }.b
}

fn main() {
    unsafe {
        let u = U::<String, Vec<u8>> { a: String::from("abcd") };

        assert_eq!(u.b.len(), 4);
        assert_eq!(u.b[0], b'a');

        let b = union_transmute::<(u8, u8), u16>((1, 1));
        assert_eq!(b, (1 << 8) + 1);

        let v: Vec<u8> = vec![1, 2, 3];
        let mut i = v.iter();
        i.next();
        let mi = MaybeItem::<std::slice::Iter<_>> { elem: i.next().unwrap() };
        assert_eq!(*mi.elem, 2);
    }
}
