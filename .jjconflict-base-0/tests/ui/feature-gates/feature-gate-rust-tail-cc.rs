#![crate_type = "lib"]

extern "tail" fn apple() {} //~ ERROR "tail" ABI is experimental

trait T {
    extern "tail" fn banana(); //~ ERROR "tail" ABI is experimental
    extern "tail" fn citrus() {} //~ ERROR "tail" ABI is experimental
}

struct S;
impl T for S {
    extern "tail" fn banana() {} //~ ERROR "tail" ABI is experimental
}

impl S {
    extern "tail" fn durian() {} //~ ERROR "tail" ABI is experimental
}

type Fig = extern "tail" fn(); //~ ERROR "tail" ABI is experimental

extern "tail" {} //~ ERROR "tail" ABI is experimental
