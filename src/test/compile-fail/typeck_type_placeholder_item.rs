// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks that it is not possible to enable global type
// inference by using the `_` type placeholder.

fn test() -> _ { 5 }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

fn test2() -> (_, _) { (5u, 5u) }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
//~^^ ERROR the type placeholder `_` is not allowed within types on item signatures.

static TEST3: _ = "test";
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

static TEST4: _ = 145u16;
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

static TEST5: (_, _) = (1, 2);
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
//~^^ ERROR the type placeholder `_` is not allowed within types on item signatures.

fn test6(_: _) { }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

fn test7(x: _) { let _x: uint = x; }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

fn test8(_f: fn() -> _) { }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

struct Test9;

impl Test9 {
    fn test9(&self) -> _ { () }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    fn test10(&self, _x : _) { }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
}

impl Clone for Test9 {
    fn clone(&self) -> _ { Test9 }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    fn clone_from(&mut self, other: _) { *self = Test9; }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
}

struct Test10 {
    a: _,
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
    b: (_, _),
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
    //~^^ ERROR the type placeholder `_` is not allowed within types on item signatures.
}

pub fn main() {
    fn fn_test() -> _ { 5 }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    fn fn_test2() -> (_, _) { (5u, 5u) }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
    //~^^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    static FN_TEST3: _ = "test";
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    static FN_TEST4: _ = 145u16;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    static FN_TEST5: (_, _) = (1, 2);
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
    //~^^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    fn fn_test6(_: _) { }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    fn fn_test7(x: _) { let _x: uint = x; }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    fn fn_test8(_f: fn() -> _) { }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

    struct FnTest9;

    impl FnTest9 {
        fn fn_test9(&self) -> _ { () }
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

        fn fn_test10(&self, _x : _) { }
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
    }

    impl Clone for FnTest9 {
        fn clone(&self) -> _ { FnTest9 }
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.

        fn clone_from(&mut self, other: _) { *self = FnTest9; }
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
    }

    struct FnTest10 {
        a: _,
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
        b: (_, _),
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures.
        //~^^ ERROR the type placeholder `_` is not allowed within types on item signatures.
    }

}
