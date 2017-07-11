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

// At time of authorship, a crate-level #![bench] with no `--test`
// will cause compilation to error unconditionally with "main function
// not found" (despite having one), similar to #[bench].
//
// (The non-crate level cases are in
// issue-43106-gating-of-builtin-attrs.rs.)

// See issue-12997-1.rs and issue-12997-2.rs to see how `#[bench]` is
// handled in "weird places" when `--test` is passed.

#![bench                   = "4100"]

fn main() { }
