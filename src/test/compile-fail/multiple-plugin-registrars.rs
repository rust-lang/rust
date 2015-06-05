// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: multiple plugin registration functions found
// error-pattern: one is here
// error-pattern: one is here
// error-pattern: aborting due to previous error

#![feature(plugin_registrar)]

// the registration function isn't typechecked yet
#[plugin_registrar]
pub fn one() {}

#[plugin_registrar]
pub fn two() {}

fn main() {}
