// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const TUP: (usize,) = (42,);

fn main() {
    let a: [isize; TUP.1];
    //~^ ERROR array length constant evaluation error: tuple index out of bounds
    //~| ERROR attempted out-of-bounds tuple index
    //~| ERROR attempted out-of-bounds tuple index
}
