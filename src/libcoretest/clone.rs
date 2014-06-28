// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn test_borrowed_clone() {
    let x = 5i;
    let y: &int = &x;
    let z: &int = (&y).clone();
    assert_eq!(*z, 5);
}

#[test]
fn test_clone_from() {
    let a = box 5i;
    let mut b = box 10i;
    b.clone_from(&a);
    assert_eq!(*b, 5);
}

#[test]
fn test_extern_fn_clone() {
    trait Empty {}
    impl Empty for int {}

    fn test_fn_a() -> f64 { 1.0 }
    fn test_fn_b<T: Empty>(x: T) -> T { x }
    fn test_fn_c(_: int, _: f64, _: int, _: int, _: int) {}

    let _ = test_fn_a.clone();
    let _ = test_fn_b::<int>.clone();
    let _ = test_fn_c.clone();
}
