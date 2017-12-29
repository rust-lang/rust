// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `/**/` was previously regarded as a doc comment because it starts with `/**` and ends with `*/`.
// This could break some internal logic that assumes the length of a doc comment is at least 5,
// leading to an ICE.

macro_rules! one_arg_macro {
    ($fmt:expr) => (print!(concat!($fmt, "\n")));
}

fn main() {
    one_arg_macro!(/**/); //~ ERROR unexpected end
}
