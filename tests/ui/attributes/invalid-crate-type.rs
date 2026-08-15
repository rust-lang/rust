// regression test for issue 11256
#![crate_type="foo"]    //~ ERROR invalid `crate_type` value
//~| NOTE `#[deny(unknown_crate_types)]` on by default
//~| HELP did you mean
//~| SUGGESTION lib

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
//~| SUGGESTION lib

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

// substring matching tests
#![crate_type="binary"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION bin

#![crate_type="library"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION lib

#![crate_type="rustlib"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION rlib

#![crate_type="dynamiclib"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION dylib

#![crate_type="dylibrary"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION dylib

#![crate_type="cdynamiclib"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION cdylib

#![crate_type="cdylibrary"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION cdylib

#![crate_type="staticlibrary"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION staticlib

#![crate_type="procedural-macro"]
//~^ ERROR invalid `crate_type` value
//~| HELP did you mean
//~| SUGGESTION proc-macro

fn main() {
    return
}
