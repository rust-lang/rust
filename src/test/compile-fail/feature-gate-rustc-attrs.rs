// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#[rustc_variance] //~ ERROR the `#[rustc_variance]` attribute is just used for rustc unit tests and will never be stable
#[rustc_error] //~ ERROR the `#[rustc_error]` attribute is just used for rustc unit tests and will never be stable
#[rustc_foo]
//~^ ERROR unless otherwise specified, attributes with the prefix `rustc_` are reserved for internal compiler diagnostics

fn main() {}
