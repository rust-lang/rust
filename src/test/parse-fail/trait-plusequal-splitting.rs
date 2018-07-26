// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only
// Fixes issue where `+` in generics weren't parsed if they were part of a `+=`.

struct Whitespace<T: Clone + = ()> { t: T }
struct TokenSplit<T: Clone +=  ()> { t: T }

fn main() {
}

FAIL //~ ERROR
