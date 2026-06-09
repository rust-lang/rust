#![feature(c_variadic)]
#![crate_type = "lib"]

// Check that `...` in closures is rejected.

const F: extern "C" fn(...) = |_: ...| {};
//~^ ERROR: unexpected `...`
//~| NOTE: only `extern "C"` and `extern "C-unwind"` functions may have a C variable argument list

fn foo() {
    let f = |...| {};
    //~^ ERROR: unexpected `...`
    //~| NOTE: not a valid pattern
    //~| NOTE: only `extern "C"` and `extern "C-unwind"` functions may have a C variable argument list

    let f = |_: ...| {};
    //~^ ERROR: unexpected `...`
    //~| NOTE: only `extern "C"` and `extern "C-unwind"` functions may have a C variable argument list
    f(1i64)
}
