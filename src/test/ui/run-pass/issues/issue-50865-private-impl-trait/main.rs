// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lib.rs

// Regression test for #50865.
// When using generics or specifying the type directly, this example
// codegens `foo` internally. However, when using a private `impl Trait`
// function which references another private item, `foo` (in this case)
// wouldn't be codegenned until main.rs used `bar`, as with impl Trait
// it is not cast to `fn()` automatically to satisfy e.g.
// `fn foo() -> fn() { ... }`.

extern crate lib;

fn main() {
    lib::bar(()); // Error won't happen if bar is called from same crate
}
