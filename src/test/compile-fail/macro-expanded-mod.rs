// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that macro-expanded non-inline modules behave correctly

macro_rules! mod_decl {
    ($i:ident) => { mod $i; }
}

mod macro_expanded_mod_helper {
    mod_decl!(foo); // This should search in the folder `macro_expanded_mod_helper`
}

fn main() {
    mod_decl!(foo); //~ ERROR Cannot declare a non-inline module inside a block
}
