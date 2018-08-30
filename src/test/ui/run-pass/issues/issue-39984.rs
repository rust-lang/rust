// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #39984.
//
// The key here is that the error type of the `Ok` call ought to be
// constrained to `String`, even though it is dead-code.

fn main() {}

fn t() -> Result<(), String> {
    return Err("".into());
    Ok(())
}
