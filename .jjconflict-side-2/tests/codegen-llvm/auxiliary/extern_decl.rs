// Auxiliary crate that exports a function and static. Both always
// evaluate to `71`. We force mutability on the static to prevent
// it from being inlined as constant.

#![crate_type = "lib"]

#[no_mangle]
pub fn extern_fn() -> u8 {
    unsafe { extern_static }
}

#[no_mangle]
pub static mut extern_static: u8 = 71;
