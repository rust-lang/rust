// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that 'custom_attributes' feature does not allow scoped attributes.

#![feature(custom_attributes)]

#[foo::bar]
//~^ ERROR scoped attribute `foo::bar` is experimental (see issue #44690) [E0658]
//~^^ ERROR an unknown tool name found in scoped attribute: `foo::bar`. [E0694]
fn main() {}
