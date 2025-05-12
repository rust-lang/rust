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
