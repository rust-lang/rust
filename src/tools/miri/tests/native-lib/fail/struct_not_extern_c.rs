// Only works on Unix targets
//@ignore-target: windows wasm
//@only-on-host

#![allow(improper_ctypes)]

pub struct PassMe {
    pub value: i32,
    pub other_value: i64,
}

extern "C" {
    fn pass_struct(s: PassMe) -> i64;
}

fn main() {
    let pass_me = PassMe { value: 42, other_value: 1337 };
    unsafe { pass_struct(pass_me) }; //~ ERROR: unsupported operation: passing a non-#[repr(C)] struct over FFI
}
