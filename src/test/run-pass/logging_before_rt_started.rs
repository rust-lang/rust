// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_LOG=std::ptr

// In issue #9487, it was realized that std::ptr was invoking the logging
// infrastructure, and when std::ptr was used during runtime initialization,
// this caused some serious problems. The problems have since been fixed, but
// this test will trigger "output during runtime initialization" to make sure
// that the bug isn't re-introduced.

pub fn main() {}
