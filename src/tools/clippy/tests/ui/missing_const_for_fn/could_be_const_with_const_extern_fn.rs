#![warn(clippy::missing_const_for_fn)]
#![feature(abi_vectorcall)]
#![feature(const_extern_fn)]

extern "C-unwind" fn c_unwind() {}
//~^ ERROR: this could be a `const fn`
extern "system" fn system() {}
//~^ ERROR: this could be a `const fn`
extern "system-unwind" fn system_unwind() {}
//~^ ERROR: this could be a `const fn`
pub extern "vectorcall" fn vector_call() {}
//~^ ERROR: this could be a `const fn`
pub extern "vectorcall-unwind" fn vector_call_unwind() {}
//~^ ERROR: this could be a `const fn`
