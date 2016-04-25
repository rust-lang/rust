// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// RFC 401 test extracted into distinct file. This is because some the
// change to suppress "derived" errors wound up suppressing this error
// message, since the fallback for `3` doesn't occur.

fn main() {
    let _ = 3 as bool;
    //~^ ERROR cannot cast as `bool`
    //~| HELP compare with zero
}
