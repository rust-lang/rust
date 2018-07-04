// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro-vis.rs
// build-aux-docs
// ignore-cross-compile

#![feature(use_extern_macros)]

#[macro_use] extern crate qwop;

// @has macro_vis/macro.some_macro.html
// @has macro_vis/index.html '//a/@href' 'macro.some_macro.html'
pub use qwop::some_macro;

// @has macro_vis/macro.renamed_macro.html
// @!has - '//pre' 'some_macro'
// @has macro_vis/index.html '//a/@href' 'macro.renamed_macro.html'
#[doc(inline)]
pub use qwop::some_macro as renamed_macro;

// @!has macro_vis/macro.other_macro.html
// @!has macro_vis/index.html '//a/@href' 'macro.other_macro.html'
// @!has - '//code' 'pub use qwop::other_macro;'
#[doc(hidden)]
pub use qwop::other_macro;

// @has macro_vis/index.html '//code' 'pub use qwop::super_macro;'
// @!has macro_vis/macro.super_macro.html
#[doc(no_inline)]
pub use qwop::super_macro;

// @has macro_vis/macro.this_is_dope.html
// @has macro_vis/index.html '//a/@href' 'macro.this_is_dope.html'
/// What it says on the tin.
#[macro_export]
macro_rules! this_is_dope {
    () => {
        println!("yo check this out");
    };
}
