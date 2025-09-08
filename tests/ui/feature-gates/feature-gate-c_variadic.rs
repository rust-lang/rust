#![crate_type="lib"]

pub unsafe extern "C" fn test(_: i32, ap: ...) { }
//~^ ERROR C-variadic functions are unstable

trait Trait {
    unsafe extern "C" fn trait_test(_: i32, ap: ...) { }
    //~^ ERROR C-variadic functions are unstable
    //~| ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
}
