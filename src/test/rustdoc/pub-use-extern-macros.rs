// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:pub-use-extern-macros.rs

#![feature(use_extern_macros)]

extern crate macros;

// @has pub_use_extern_macros/macro.bar.html
// @!has pub_use_extern_macros/index.html '//code' 'pub use macros::bar;'
pub use macros::bar;

// @has pub_use_extern_macros/macro.baz.html
// @!has pub_use_extern_macros/index.html '//code' 'pub use macros::baz;'
#[doc(inline)]
pub use macros::baz;

// @has pub_use_extern_macros/macro.quux.html
// @!has pub_use_extern_macros/index.html '//code' 'pub use macros::quux;'
#[doc(hidden)]
pub use macros::quux;
