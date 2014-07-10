// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

__register_diagnostic!(E0001)
//~^ ERROR macro undefined: '__register_diagnostic!'

fn main() {
    __diagnostic_used!(E0001);
    //~^ ERROR macro undefined: '__diagnostic_used!'
}

__build_diagnostic_array!(DIAGNOSTICS)
//~^ ERROR macro undefined: '__build_diagnostic_array!'
