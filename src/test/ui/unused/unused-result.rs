// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![deny(unused_results, unused_must_use)]
//~^ NOTE: lint level defined here
//~| NOTE: lint level defined here

#[must_use]
enum MustUse { Test }

#[must_use = "some message"]
enum MustUseMsg { Test2 }

fn foo<T>() -> T { panic!() }

fn bar() -> isize { return foo::<isize>(); }
fn baz() -> MustUse { return foo::<MustUse>(); }
fn qux() -> MustUseMsg { return foo::<MustUseMsg>(); }

#[allow(unused_results)]
fn test() {
    foo::<isize>();
    foo::<MustUse>(); //~ ERROR: unused `MustUse` which must be used
    foo::<MustUseMsg>(); //~ ERROR: unused `MustUseMsg` which must be used
    //~^ NOTE: some message
}

#[allow(unused_results, unused_must_use)]
fn test2() {
    foo::<isize>();
    foo::<MustUse>();
    foo::<MustUseMsg>();
}

fn main() {
    foo::<isize>(); //~ ERROR: unused result
    foo::<MustUse>(); //~ ERROR: unused `MustUse` which must be used
    foo::<MustUseMsg>(); //~ ERROR: unused `MustUseMsg` which must be used
    //~^ NOTE: some message

    let _ = foo::<isize>();
    let _ = foo::<MustUse>();
    let _ = foo::<MustUseMsg>();
}
