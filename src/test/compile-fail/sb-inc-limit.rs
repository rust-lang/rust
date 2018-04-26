// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test to see how file sandboxing is working. This blocks all includes.
// compile-flags:--include-prefix {{src-base}}/sb-fixtures/a

fn main() {
    let _ = include_str!("sb-fixtures/a/a.in");
    let _ = include_str!("sb-fixtures/b/b.in"); //~ERROR: path does not have a valid prefix
}
