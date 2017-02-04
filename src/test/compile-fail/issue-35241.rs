// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo {
    Bar(u32),
}

fn get_foo() -> Foo {
    Foo::Bar
    //~^ ERROR E0308
    //~| HELP `Foo::Bar` is a function - maybe try calling it?
}

fn get_u32() -> u32 {
    Foo::Bar
    //~^ ERROR E0308
}

fn abort_2() {
    abort
    //~^ ERROR E0308
    //~| HELP `abort` is a function - maybe try calling it?
}

fn abort() -> ! { panic!() }

fn call(f: fn() -> u32) -> u32 {
    f
    //~^ ERROR E0308
    //~| HELP found a function - maybe try calling it?
}

fn main() {}