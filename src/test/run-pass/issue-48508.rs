// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #48508:
//
// Confusion between global and local file offsets caused incorrect handling of multibyte character
// spans when compiling multiple files. One visible effect was an ICE generating debug information
// when a multibyte character is at the end of a scope. The problematic code is actually in
// issue-48508-aux.rs

// compile-flags:-g

#![feature(non_ascii_idents)]

#[path = "issue-48508-aux.rs"]
mod other_file;

fn main() {
    other_file::other();
}
