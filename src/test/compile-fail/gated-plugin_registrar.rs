// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-plugin_registrar

// Test that `#[plugin_registrar]` attribute is gated by `plugin_registrar`
// feature gate.

// the registration function isn't typechecked yet
#[plugin_registrar]
pub fn registrar() {}
//~^ ERROR compiler plugins are experimental
fn main() {}
