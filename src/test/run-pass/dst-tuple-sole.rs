// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// As dst-tuple.rs, but the unsized field is the only field in the tuple.


#![feature(unsized_tuple_coercion)]

type Fat<T: ?Sized> = (T,);

// x is a fat pointer
fn foo(x: &Fat<[isize]>) {
    let y = &x.0;
    assert_eq!(x.0.len(), 3);
    assert_eq!(y[0], 1);
    assert_eq!(x.0[1], 2);
}

fn foo2<T:ToBar>(x: &Fat<[T]>) {
    let y = &x.0;
    let bar = Bar;
    assert_eq!(x.0.len(), 3);
    assert_eq!(y[0].to_bar(), bar);
    assert_eq!(x.0[1].to_bar(), bar);
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Bar;

trait ToBar {
    fn to_bar(&self) -> Bar;
}

impl ToBar for Bar {
    fn to_bar(&self) -> Bar {
        *self
    }
}

pub fn main() {
    // With a vec of ints.
    let f1 = ([1, 2, 3],);
    foo(&f1);
    let f2 = &f1;
    foo(f2);
    let f3: &Fat<[isize]> = f2;
    foo(f3);
    let f4: &Fat<[isize]> = &f1;
    foo(f4);
    let f5: &Fat<[isize]> = &([1, 2, 3],);
    foo(f5);

    // With a vec of Bars.
    let bar = Bar;
    let f1 = ([bar, bar, bar],);
    foo2(&f1);
    let f2 = &f1;
    foo2(f2);
    let f3: &Fat<[Bar]> = f2;
    foo2(f3);
    let f4: &Fat<[Bar]> = &f1;
    foo2(f4);
    let f5: &Fat<[Bar]> = &([bar, bar, bar],);
    foo2(f5);

    // Assignment.
    let f5: &mut Fat<[isize]> = &mut ([1, 2, 3],);
    f5.0[1] = 34;
    assert_eq!(f5.0[0], 1);
    assert_eq!(f5.0[1], 34);
    assert_eq!(f5.0[2], 3);

    // Zero size vec.
    let f5: &Fat<[isize]> = &([],);
    assert!(f5.0.is_empty());
    let f5: &Fat<[Bar]> = &([],);
    assert!(f5.0.is_empty());
}
