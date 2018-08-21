// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ambiguity between a `macro_rules` macro and a non-existent import recovered as `Def::Err`

macro_rules! mac { () => () }

mod m {
    use nonexistent_module::mac; //~ ERROR unresolved import `nonexistent_module`

    mac!(); //~ ERROR `mac` is ambiguous
}

fn main() {}
