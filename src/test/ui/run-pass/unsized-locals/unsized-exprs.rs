// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

#![feature(unsized_tuple_coercion, unsized_locals)]

struct A<X: ?Sized>(X);

fn udrop<T: ?Sized>(_x: T) {}
fn foo() -> Box<[u8]> {
    Box::new(*b"foo")
}
fn tfoo() -> Box<(i32, [u8])> {
    Box::new((42, *b"foo"))
}
fn afoo() -> Box<A<[u8]>> {
    Box::new(A(*b"foo"))
}

impl std::ops::Add<i32> for A<[u8]> {
    type Output = ();
    fn add(self, _rhs: i32) -> Self::Output {}
}

fn main() {
    udrop::<[u8]>(loop {
        break *foo();
    });
    udrop::<[u8]>(if true {
        *foo()
    } else {
        *foo()
    });
    udrop::<[u8]>({*foo()});
    #[allow(unused_parens)]
    udrop::<[u8]>((*foo()));
    udrop::<[u8]>((*tfoo()).1);
    *afoo() + 42;
}
