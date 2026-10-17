#![crate_type = "lib"]

extern "rust-preserve-none" fn apple() {} //~ ERROR "rust-preserve-none" ABI is experimental

trait T {
    extern "rust-preserve-none" fn banana(); //~ ERROR "rust-preserve-none" ABI is experimental
    extern "rust-preserve-none" fn citrus() {} //~ ERROR "rust-preserve-none" ABI is experimental
}

struct S;
impl T for S {
    extern "rust-preserve-none" fn banana() {} //~ ERROR "rust-preserve-none" ABI is experimental
}

impl S {
    extern "rust-preserve-none" fn durian() {} //~ ERROR "rust-preserve-none" ABI is experimental
}

type Fig = extern "rust-preserve-none" fn(); //~ ERROR "rust-preserve-none" ABI is experimental

extern "rust-preserve-none" {} //~ ERROR "rust-preserve-none" ABI is experimental
