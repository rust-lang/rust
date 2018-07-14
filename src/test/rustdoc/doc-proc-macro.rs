// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #52129: ICE when trying to document the `quote` proc-macro from proc_macro

// As of this writing, we don't currently attempt to document proc-macros. However, we shouldn't
// crash when we try.

extern crate proc_macro;

pub use proc_macro::*;
