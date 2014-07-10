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

// NOTE: remove after next snapshot
#[cfg(stage0)]
#[macro_export]
macro_rules! __register_diagnostic(
    ($code:tt, $description:tt) => ();
    ($code:tt) => ()
)

#[macro_export]
macro_rules! register_diagnostic(
    ($code:tt, $description:tt) => (__register_diagnostic!($code, $description));
    ($code:tt) => (__register_diagnostic!($code))
)

// NOTE: remove after next snapshot
#[cfg(stage0)]
#[macro_export]
macro_rules! __build_diagnostic_array(
    ($name:ident) => {
        pub static $name: [(&'static str, &'static str), ..0] = [];
    }
)

// NOTE: remove after next snapshot
#[cfg(stage0)]
#[macro_export]
macro_rules! __diagnostic_used(
    ($code:ident) => {
        ()
    }
)

#[macro_export]
macro_rules! span_err(
    ($session:expr, $span:expr, $code:ident, $($arg:expr),*) => ({
        __diagnostic_used!($code);
        ($session).span_err_with_code($span, format!($($arg),*).as_slice(), stringify!($code))
    })
)
