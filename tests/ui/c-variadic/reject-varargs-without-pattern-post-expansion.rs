#![crate_type = "lib"]
#![warn(varargs_without_pattern)]

// Test that we reject a bare `...` without a pattern post-expansion in function definitons and
// trait method declarations. On foreign function declarations it is allowed.
//
// We have the `varargs_without_pattern` FCW for this idiom, with the intent to eventually also
// reject this idiom pre-expansion.

// Bare `...` is allowed in extern blocks.
extern "C" {
    fn g(...);
}

// When the `...` argument does not make it past expansion, that only lints.
macro_rules! discard_item {
    ($item:item) => {};
}

discard_item! {
    unsafe extern "C" fn f(...) -> i32 {
        //~^ WARN missing pattern for `...` argument
        //~| WARN this was previously accepted by the compiler but is being phased out
        0
    }
}

// But when it does make it post-expansion, that is a hard error.
macro_rules! identity_item {
    ($item:item) => {
        $item
    };
}

identity_item! {
    unsafe extern "C" fn f(...) {}
        //~^ ERROR missing pattern for `...` argument
        //~| WARN missing pattern for `...` argument
        //~| WARN this was previously accepted by the compiler but is being phased out
        //~| WARN missing pattern for `...` argument
        //~| WARN this was previously accepted by the compiler but is being phased out
}

trait T {
    identity_item! {
        unsafe extern "C" fn f(...);
        //~^ ERROR missing pattern for `...` argument
        //~| WARN missing pattern for `...` argument
        //~| WARN this was previously accepted by the compiler but is being phased out
        //~| WARN missing pattern for `...` argument
        //~| WARN this was previously accepted by the compiler but is being phased out
        //~| WARN anonymous_parameters
        //~| WARN this is accepted in the current edition (Rust 2015)
    }
}
