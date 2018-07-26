// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Tests the formatting of the feature-gate errors for non_modrs_mods
//
// gate-test-non_modrs_mods
// ignore-windows
// ignore-pretty issue #37195
pub mod modrs_mod;
pub mod foors_mod;

#[path = "some_crazy_attr_mod_dir/arbitrary_name.rs"]
pub mod attr_mod;

pub fn main() {
    modrs_mod::inner_modrs_mod::innest::foo();
    modrs_mod::inner_foors_mod::innest::foo();
    foors_mod::inner_modrs_mod::innest::foo();
    foors_mod::inner_foors_mod::innest::foo();
    attr_mod::inner_modrs_mod::innest::foo();
}
