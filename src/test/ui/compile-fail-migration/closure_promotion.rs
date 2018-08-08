// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(const_err)]

// nll successfully compiles this. It is a bug.
// See https://github.com/rust-lang/rust/issues/52384
fn main() {
    let x: &'static _ = &|| { let z = 3; z }; //~ ERROR does not live long enough
}
