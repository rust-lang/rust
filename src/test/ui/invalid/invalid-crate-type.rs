// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// regression test for issue 11256
#![crate_type="foo"]    //~ ERROR invalid `crate_type` value

// Tests for suggestions (#53958)

#![crate_type="statoclib"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION staticlib

#![crate_type="procmacro"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION proc-macro

#![crate_type="static-lib"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION staticlib

#![crate_type="drylib"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION dylib

#![crate_type="dlib"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION rlib

#![crate_type="lob"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION lib

#![crate_type="bon"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION bin

#![crate_type="cdalib"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION cdylib

fn main() {
    return
}
