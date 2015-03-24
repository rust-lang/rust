// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:crate_with_invalid_spans.rs

// pretty-expanded FIXME #23616

extern crate crate_with_invalid_spans;

fn main() {
    // The AST of `exported_generic` stored in crate_with_invalid_spans's
    // metadata should contain an invalid span where span.lo > span.hi.
    // Let's make sure the compiler doesn't crash when encountering this.
    let _ = crate_with_invalid_spans::exported_generic(32u32, 7u32);
}
