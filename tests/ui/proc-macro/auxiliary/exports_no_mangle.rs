//@ force-host
//@ no-prefer-dynamic
#![crate_type="lib"]

// Issue 111888: this crate (1.) is imported by a proc-macro crate and (2.)
// exports a no_mangle function; that combination of acts was broken for some
// period of time. See further discussion in the test file that imports this
// crate.

#[no_mangle]
pub fn some_no_mangle_function() { }
