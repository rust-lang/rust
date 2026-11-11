#![crate_type = "lib"]

// Check that `...` in closures is rejected.

const F: extern "C" fn(...) = |_: ...| {};
//~^ ERROR: C-variadic type `...` is not allowed in this context
//~| NOTE: `...` is only allowed as the type of function parameters

fn foo() {
    let f = |...| {};
    //~^ ERROR: unexpected `...`
    //~| NOTE: not a valid pattern
    //~| NOTE: `...` is only allowed as the type of function parameters

    let f = |_: ...| {};
    //~^ ERROR: C-variadic type `...` is not allowed in this context
    //~| NOTE: `...` is only allowed as the type of function parameters
    f(1i64)
}
