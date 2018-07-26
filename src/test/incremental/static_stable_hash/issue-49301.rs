// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// https://github.com/rust-lang/rust/issues/49081

// revisions:rpass1 rpass2

#[cfg(rpass1)]
pub static A: &str = "hello";
#[cfg(rpass2)]
pub static A: &str = "xxxxx";

#[cfg(rpass1)]
fn main() {
    assert_eq!(A, "hello");
}

#[cfg(rpass2)]
fn main() {
    assert_eq!(A, "xxxxx");
}
