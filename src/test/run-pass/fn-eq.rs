// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn one(x: int) -> int { x + 1 }
fn two(x: int) -> int { x + 2 }

fn generic<T>(x: T) -> T { x }

extern fn foo(_x: i32) {}
extern {
    fn exit(n: i32);
}

fn main() {
    assert!(one == one);
    assert!(one != two);
    assert!(one != generic::<int>);

    assert!(two != one);
    assert!(two == two);
    assert!(two != generic::<int>);

    assert!(generic::<int> != one);
    assert!(generic::<int> != two);
    assert!(generic::<int> == generic::<int>);

    assert!(foo == foo);
    assert!(foo != exit);
    assert!(exit != foo);
    assert!(exit == exit);

    let bar: unsafe extern "C" fn(i32) = foo;
    assert!(bar == foo);
    assert!(bar != exit);
    assert!(exit != bar);
}
