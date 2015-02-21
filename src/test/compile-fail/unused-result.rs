// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_results, unused_must_use)]
#![allow(dead_code)]

#[must_use]
enum MustUse { Test }

#[must_use = "some message"]
enum MustUseMsg { Test2 }

fn foo<T>() -> T { panic!() }

fn bar() -> isize { return foo::<isize>(); }
fn baz() -> MustUse { return foo::<MustUse>(); }
fn qux() -> MustUseMsg { return foo::<MustUseMsg>(); }

#[must_use]
fn func() -> bool { true }
#[must_use = "some message"]
fn func_msg() -> i32 { 1 }

impl MustUse {
    #[must_use]
    fn method(&self) -> f64 { 0.0 }
    #[must_use = "some message"]
    fn method_msg(&self) -> &str { "foo" }
}

#[allow(unused_results)]
fn test() {
    foo::<isize>();
    foo::<MustUse>(); //~ ERROR: unused result which must be used
    foo::<MustUseMsg>(); //~ ERROR: unused result which must be used: some message
    func(); //~ ERROR: unused result which must be used
    func_msg(); //~ ERROR: unused result which must be used: some message

    MustUse::Test.method(); //~ ERROR: unused result which must be used
    MustUse::Test.method_msg(); //~ ERROR: unused result which must be used: some message
}

#[allow(unused_results, unused_must_use)]
fn test2() {
    foo::<isize>();
    foo::<MustUse>();
    foo::<MustUseMsg>();
    func();
    func_msg();
    MustUse::Test.method();
    MustUse::Test.method_msg();
}

fn main() {
    foo::<isize>(); //~ ERROR: unused result
    foo::<MustUse>(); //~ ERROR: unused result which must be used
    foo::<MustUseMsg>(); //~ ERROR: unused result which must be used: some message
    func(); //~ ERROR: unused result which must be used
    func_msg(); //~ ERROR: unused result which must be used: some message
    MustUse::Test.method(); //~ ERROR: unused result which must be used
    MustUse::Test.method_msg(); //~ ERROR: unused result which must be used: some message

    let _ = foo::<isize>();
    let _ = foo::<MustUse>();
    let _ = foo::<MustUseMsg>();
    let _ = func();
    let _ = func_msg();
    let _ = MustUse::Test.method();
    let _ = MustUse::Test.method_msg();
}
