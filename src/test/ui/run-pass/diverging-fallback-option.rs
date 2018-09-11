// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(warnings)]

// Here the type of `c` is `Option<?T>`, where `?T` is unconstrained.
// Because there is data-flow from the `{ return; }` block, which
// diverges and hence has type `!`, into `c`, we will default `?T` to
// `!`, and hence this code compiles rather than failing and requiring
// a type annotation.

fn main() {
    let c = Some({ return; });
    c.unwrap();
}
