#![warn(clippy::missing_const_for_fn)]
#![allow(unsupported_calling_conventions)]
#![feature(const_extern_fn)]

extern "C-unwind" fn c_unwind() {}
//~^ ERROR: this could be a `const fn`
extern "system" fn system() {}
//~^ ERROR: this could be a `const fn`
extern "system-unwind" fn system_unwind() {}
//~^ ERROR: this could be a `const fn`
pub extern "stdcall" fn std_call() {}
//~^ ERROR: this could be a `const fn`
pub extern "stdcall-unwind" fn std_call_unwind() {}
//~^ ERROR: this could be a `const fn`
