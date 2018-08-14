// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a companion to the similarly-named test in run-pass.
//
// It tests macros that unavoidably produce compile errors.

fn compile_error() {
    compile_error!("lel"); //~ ERROR lel
    compile_error!("lel",); //~ ERROR lel
}

fn main() {}
