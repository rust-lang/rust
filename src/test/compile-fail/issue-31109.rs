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
    // FIXME(#31407) this error should go away, but in the meantime we test that it
    // is accompanied by a somewhat useful error message.
    let _: f64 = 1234567890123456789012345678901234567890e-340;
    //~^ ERROR constant evaluation error
    //~| unimplemented constant expression: could not evaluate float literal
}
