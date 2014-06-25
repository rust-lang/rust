// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn foo(&self, mut x: int) -> int {
        let val = x;
        x = 37 * x;
        val + x
    }
}

struct X;
impl Foo for X {}

pub fn main() {
    let (a, mut b) = (23i, 4i);
    assert_eq!(a, 23);
    assert_eq!(b, 4);
    b = a + b;
    assert_eq!(b, 27);


    assert_eq!(X.foo(2), 76);

    enum Bar {
       Foo(int),
       Baz(f32, u8)
    }

    let (x, mut y) = (32i, Foo(21));

    match x {
        mut z @ 32 => {
            assert_eq!(z, 32);
            z = 34;
            assert_eq!(z, 34);
        }
        _ => {}
    }

    check_bar(&y);
    y = Baz(10.0, 3);
    check_bar(&y);

    fn check_bar(y: &Bar) {
        match y {
            &Foo(a) => {
                assert_eq!(a, 21);
            }
            &Baz(a, b) => {
                assert_eq!(a, 10.0);
                assert_eq!(b, 3);
            }
        }
    }

    fn foo1((x, mut y): (f64, int), mut z: int) -> int {
        y = 2 * 6;
        z = y + (x as int);
        y - z
    }

    struct A {
        x: int
    }
    let A { x: mut x } = A { x: 10 };
    assert_eq!(x, 10);
    x = 30;
    assert_eq!(x, 30);

    (|A { x: mut t }: A| { t = t+1; t })(A { x: 34 });

}
