// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern {
    fn sqrt<T>(f: T) -> T;
    //~^ ERROR foreign items may not have type parameters [E0044]
    //~| HELP use specialization instead of type parameters by replacing them with concrete types
    //~| NOTE can't have type parameters
}

fn main() {
}
