#![crate_type = "lib"]

extern "rust-cold" fn fu() {} //~ ERROR "rust-cold" ABI is experimental

trait T {
    extern "rust-cold" fn mu(); //~ ERROR "rust-cold" ABI is experimental
    extern "rust-cold" fn dmu() {} //~ ERROR "rust-cold" ABI is experimental
}

struct S;
impl T for S {
    extern "rust-cold" fn mu() {} //~ ERROR "rust-cold" ABI is experimental
}

impl S {
    extern "rust-cold" fn imu() {} //~ ERROR "rust-cold" ABI is experimental
}

type TAU = extern "rust-cold" fn(); //~ ERROR "rust-cold" ABI is experimental

extern "rust-cold" {} //~ ERROR "rust-cold" ABI is experimental
