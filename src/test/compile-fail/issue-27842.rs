// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let tup = (0, 1, 2);
    // the case where we show a suggestion
    let _ = tup[0];
    //~^ ERROR cannot index a value of type
    //~| HELP to access tuple elements, use tuple indexing syntax as shown
    //~| SUGGESTION let _ = tup.0

    // the case where we show just a general hint
    let i = 0_usize;
    let _ = tup[i];
    //~^ ERROR cannot index a value of type
    //~| HELP to access tuple elements, use tuple indexing syntax (e.g. `tuple.0`)
}
