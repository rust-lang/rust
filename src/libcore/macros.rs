// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]

/// Entry point of failure, for details, see std::macros
#[macro_export]
macro_rules! fail(
    () => (
        fail!("explicit failure")
    );
    ($msg:expr) => (
        ::failure::begin_unwind($msg, file!(), line!())
    );
)

/// Runtime assertion, for details see std::macros
#[macro_export]
macro_rules! assert(
    ($cond:expr) => (
        if !$cond {
            fail!(concat!("assertion failed: ", stringify!($cond)))
        }
    );
)

/// Runtime assertion, disableable at compile time
#[macro_export]
macro_rules! debug_assert(
    ($($arg:tt)*) => (if cfg!(not(ndebug)) { assert!($($arg)*); })
)
