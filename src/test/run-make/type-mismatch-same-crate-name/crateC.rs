// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This tests the extra note reported when a type error deals with
// seemingly identical types.
// The main use case of this error is when there are two crates
// (generally different versions of the same crate) with the same name
// causing a type mismatch.

// The test is nearly the same as the one in
// compile-fail/type-mismatch-same-crate-name.rs
// but deals with the case where one of the crates
// is only introduced as an indirect dependency.
// and the type is accessed via a re-export.
// This is similar to how the error can be introduced
// when using cargo's automatic dependency resolution.

extern crate crateA;

fn main() {
    let foo2 = crateA::Foo;
    let bar2 = crateA::bar();
    {
        extern crate crateB;
        crateB::try_foo(foo2);
        crateB::try_bar(bar2);
    }
}
