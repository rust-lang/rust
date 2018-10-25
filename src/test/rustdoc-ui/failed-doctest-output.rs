// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #51162: A failed doctest was not printing its stdout/stderr
// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// failure-status: 101
// rustc-env:RUST_BACKTRACE=0

// doctest fails at runtime
/// ```
/// panic!("oh no");
/// ```
pub struct SomeStruct;

// doctest fails at compile time
/// ```
/// no
/// ```
pub struct OtherStruct;
