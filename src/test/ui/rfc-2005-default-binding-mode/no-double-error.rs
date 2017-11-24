// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Without caching type lookups in FnCtxt.resolve_ty_and_def_ufcs
// the error below would be reported twice (once when checking
// for a non-ref pattern, once when processing the pattern).

fn main() {
    let foo = 22;
    match foo {
        u32::XXX => { } //~ ERROR no associated item named
        _ => { }
    }
}
