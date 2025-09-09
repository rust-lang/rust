// For now C-variadic arguments in trait methods are rejected, though we aim to lift this
// restriction in the future. In particular we need to think about the interaction with
// `dyn Trait` and the `ReifyShim`s that it may generate for methods.
#![feature(c_variadic)]
#![crate_type = "lib"]
struct S;

impl S {
    unsafe extern "C" fn associated_function(mut ap: ...) -> i32 {
        unsafe { ap.arg() }
    }

    unsafe extern "C" fn method(&self, mut ap: ...) -> i32 {
        unsafe { ap.arg() }
    }
}

trait T {
    unsafe extern "C" fn trait_associated_function(mut ap: ...) -> i32 {
        //~^ ERROR: associated functions cannot have a C variable argument list
        unsafe { ap.arg() }
    }

    unsafe extern "C" fn trait_method(&self, mut ap: ...) -> i32 {
        //~^ ERROR: associated functions cannot have a C variable argument list
        unsafe { ap.arg() }
    }
}

impl T for S {}

fn main() {
    unsafe {
        assert_eq!(S::associated_function(32), 32);
        assert_eq!(S.method(32), 32);

        assert_eq!(S::trait_associated_function(32), 32);
        assert_eq!(S.trait_method(32), 32);
    }
}
