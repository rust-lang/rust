// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::local_data;

// check that the local data keys are private by default.

mod bar {
    local_data_key!(baz: float)
}

fn main() {
    local_data::set(bar::baz, -10.0);
    //~^ ERROR unresolved name `bar::baz`
}
