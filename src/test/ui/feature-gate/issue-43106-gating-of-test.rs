// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: main function not found

// At time of authorship, crate-level #[test] attribute with no
// `--test` signals unconditional error complaining of missing main
// function (despite having one), similar to #[bench].
//
// (The non-crate level cases are in
// issue-43106-gating-of-builtin-attrs.rs.)

#![test                    = "4200"]

fn main() { }
