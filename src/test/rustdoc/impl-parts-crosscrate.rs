// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:rustdoc-impl-parts-crosscrate.rs
// ignore-cross-compile

#![feature(optin_builtin_traits)]

extern crate rustdoc_impl_parts_crosscrate;

pub struct Bar<T> { t: T }

// The output file is html embedded in javascript, so the html tags
// aren't stripped by the processing script and we can't check for the
// full impl string.  Instead, just make sure something from each part
// is mentioned.

// @has implementors/rustdoc_impl_parts_crosscrate/trait.AnOibit.js Bar
// @has - Send
// @has - !AnOibit
// @has - Copy
impl<T: Send> !rustdoc_impl_parts_crosscrate::AnOibit for Bar<T>
    where T: Copy {}
