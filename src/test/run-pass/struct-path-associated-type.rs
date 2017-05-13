// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(more_struct_aliases)]

struct S<T, U = u16> {
    a: T,
    b: U,
}

trait Tr {
    type A;
}
impl Tr for u8 {
    type A = S<u8, u16>;
}

fn f<T: Tr<A = S<u8>>>() {
    let s = T::A { a: 0, b: 1 };
    match s {
        T::A { a, b } => {
            assert_eq!(a, 0);
            assert_eq!(b, 1);
        }
    }
}

fn main() {
    f::<u8>();
}
