//@ check-fail
//@ run-rustfix
//@ revisions: e2015 e2018
//@[e2015] edition: 2015
//@[e2018] edition: 2018
#![crate_type = "lib"]
#![deny(varargs_without_pattern)]

#[cfg(false)]
mod module {
    unsafe extern "C" fn f(...) {
    //~^ ERROR missing pattern for `...` argument
    //~| WARN this was previously accepted by the compiler
        unsafe extern "C" fn f(...) {}
        //~^ ERROR missing pattern for `...` argument
        //~| WARN this was previously accepted by the compiler
    }

    impl A {
        unsafe extern "C" fn f(...) {}
        //~^ ERROR missing pattern for `...` argument
        //~| WARN this was previously accepted by the compiler
    }

    trait A {
        unsafe extern "C" fn f(...) {}
        //[e2018]~^ ERROR missing pattern for `...` argument
        //[e2018]~| WARN this was previously accepted by the compiler
    }

    unsafe extern "C" {
        fn f(...); // no error
    }

    type A = unsafe extern "C" fn(...); // no error
}
