// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--test

// comments, both doc comments and regular ones, used to trick rustdoc's doctest parser into
// thinking that everything after it was part of the regular program. combined with the libsyntax
// parser loop failing to detect the manual main function, it would wrap everything in `fn main`,
// which would cause the doctest to fail as the "extern crate" declaration was no longer valid.
// oddly enough, it would pass in 2018 if a crate was in the extern prelude. see
// https://github.com/rust-lang/rust/issues/56727

//! ```
//! // crate: proc-macro-test
//! //! this is a test
//!
//! // used to pull in proc-macro specific items
//! extern crate proc_macro;
//!
//! use proc_macro::TokenStream;
//!
//! # fn main() {}
//! ```
