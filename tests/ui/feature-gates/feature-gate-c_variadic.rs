#![crate_type = "lib"]

pub unsafe extern "C" fn test(_: i32, ap: ...) {}
//~^ ERROR C-variadic functions are unstable

trait Trait {
    unsafe extern "C" fn trait_test(_: i32, ap: ...) {}
    //~^ ERROR C-variadic functions are unstable
}
