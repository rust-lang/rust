// A rust c-variadic function definition must use `_: ...`, not `...`.
#![feature(c_variadic)]
#![crate_type = "lib"]

const F: extern "C" fn(...) = |_: ...| {};
//~^ ERROR C-variadic type `...` may not be nested inside another type

fn foo() {
    let f = |...| {};
    //~^ ERROR: `..` patterns are not allowed here
    //~| ERROR: unexpected `...`

    let f = |_: ...| {};
    //~^ ERROR C-variadic type `...` may not be nested inside another type
    f(1i64)
}
