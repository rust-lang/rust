// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test to see how environment sandboxing is working
// compile-flags:--include-prefix {{src-base}}/sb-fixtures/a
// compile-flags:--include-prefix {{src-base}}/sb-fixtures/b/b.in

fn main() {
    assert_eq!(include_str!("sb-fixtures/a/a.in"), "File A\n");
    assert_eq!(include_str!("sb-fixtures/b/b.in"), "File B\n");
}
