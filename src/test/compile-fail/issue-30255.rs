// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Test that lifetime elision error messages correctly omit parameters
// with no elided lifetimes

struct S<'a> {
    field: &'a i32,
}

fn f(a: &S, b: i32) -> &i32 {
//~^ ERROR missing lifetime specifier [E0106]
//~^^ HELP does not say which one of `a`'s 2 elided lifetimes it is borrowed from
    panic!();
}

fn g(a: &S, b: bool, c: &i32) -> &i32 {
//~^ ERROR missing lifetime specifier [E0106]
//~^^ HELP does not say whether it is borrowed from one of `a`'s 2 elided lifetimes or `c`
    panic!();
}

fn h(a: &bool, b: bool, c: &S, d: &i32) -> &i32 {
//~^ ERROR missing lifetime specifier [E0106]
//~^^ HELP does not say whether it is borrowed from `a`, one of `c`'s 2 elided lifetimes, or `d`
    panic!();
}

