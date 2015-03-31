// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    0b121; //~ ERROR invalid digit for a base 2 literal
    0b10_10301; //~ ERROR invalid digit for a base 2 literal
    0b30; //~ ERROR invalid digit for a base 2 literal
    0b41; //~ ERROR invalid digit for a base 2 literal
    0b5; //~ ERROR invalid digit for a base 2 literal
    0b6; //~ ERROR invalid digit for a base 2 literal
    0b7; //~ ERROR invalid digit for a base 2 literal
    0b8; //~ ERROR invalid digit for a base 2 literal
    0b9; //~ ERROR invalid digit for a base 2 literal
}
