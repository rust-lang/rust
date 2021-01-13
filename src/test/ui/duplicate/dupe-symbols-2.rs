// build-fail

//
#![crate_type="rlib"]
#![allow(warnings)]

pub mod a {
    #[no_mangle]
    pub extern "C" fn fail() {
    }
}

pub mod b {
    #[no_mangle]
    pub extern "C" fn fail() {
    //~^ symbol `fail` is already defined
    }
}
