#![warn(unused_attributes)]

// Tests that placing the #[no_mangle] attribute on a foreign fn or static emits
// a specialized warning.
// The previous warning only talks about a "function or static" but foreign fns/statics
// are also not allowed to have #[no_mangle]

//@ build-pass

extern "C" {
    #[no_mangle]
    //~^ WARNING attribute cannot be used on
    //~| WARN previously accepted
    pub static FOO: u8;

    #[no_mangle]
    //~^ WARNING attribute cannot be used on
    //~| WARN previously accepted
    pub fn bar();
}

fn no_new_warn() {
    // Should emit the generic "not a function or static" warning
    #[no_mangle]
    //~^ WARNING attribute cannot be used on
    //~| WARN previously accepted
    let x = 0_u8;
}

fn main() {}
