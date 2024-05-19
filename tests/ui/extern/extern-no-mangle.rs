#![warn(unused_attributes)]

// Tests that placing the #[no_mangle] attribute on a foreign fn or static emits
// a specialized warning.
// The previous warning only talks about a "function or static" but foreign fns/statics
// are also not allowed to have #[no_mangle]

//@ build-pass

extern "C" {
    #[no_mangle]
    //~^ WARNING `#[no_mangle]` has no effect on a foreign static
    //~^^ WARNING this was previously accepted by the compiler
    pub static FOO: u8;

    #[no_mangle]
    //~^ WARNING `#[no_mangle]` has no effect on a foreign function
    //~^^ WARNING this was previously accepted by the compiler
    pub fn bar();
}

fn no_new_warn() {
    // Should emit the generic "not a function or static" warning
    #[no_mangle]
    //~^ WARNING attribute should be applied to a free function, impl method or static
    //~^^ WARNING this was previously accepted by the compiler
    let x = 0_u8;
}

fn main() {}
