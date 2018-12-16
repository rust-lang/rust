// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test just shows that a crate-level `#![deprecated]` does not
// signal a warning or error. (This file sits on its own because a
// crate-level `#![deprecated]` causes all that crate's item
// definitions to be deprecated, which is a pain to work with.)
//
// (For non-crate-level cases, see issue-43106-gating-of-builtin-attrs.rs)

// compile-pass
// skip-codegen
#![allow(dead_code)]
#![deprecated]

fn main() {
    println!("Hello World");
}
