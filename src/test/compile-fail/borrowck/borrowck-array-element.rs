// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck-mir

fn use_x(_: usize) -> bool { true }

fn main() {
}

fn foo() {
    let mut v = [1, 2, 3];
    let p = &v[0];
    if true {
        use_x(*p);
    } else {
        use_x(22);
    }
    v[0] += 1; //[ast]~ ERROR cannot assign to `v[..]` because it is borrowed
    //[mir]~^ cannot assign to `v[..]` because it is borrowed (Ast)
    //[mir]~| cannot assign to `v[..]` because it is borrowed (Mir)
}

fn bar() {
    let mut v = [[1]];
    let p = &v[0][0];
    if true {
        use_x(*p);
    } else {
        use_x(22);
    }
    v[0][0] += 1; //[ast]~ ERROR cannot assign to `v[..][..]` because it is borrowed
    //[mir]~^ cannot assign to `v[..][..]` because it is borrowed (Ast)
    //[mir]~| cannot assign to `v[..][..]` because it is borrowed (Mir)
}

fn baz() {
    struct S<T> { x: T, y: T, }
    let mut v = [S { x: 1, y: 2 },
                 S { x: 3, y: 4 },
                 S { x: 5, y: 6 }];
    let p = &v[0].x;
    if true {
        use_x(*p);
    } else {
        use_x(22);
    }
    v[0].x += 1; //[ast]~ ERROR cannot assign to `v[..].x` because it is borrowed
    //[mir]~^ cannot assign to `v[..].x` because it is borrowed (Ast)
    //[mir]~| cannot assign to `v[..].x` because it is borrowed (Mir)
}
