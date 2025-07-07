//@ check-fail
//@ revisions: e2015 e2018
//@[e2015] edition: 2015
//@[e2018] edition: 2018
#![crate_type = "lib"]

#[cfg(false)]
mod module {
    unsafe extern "C" fn f(...) {
    //~^ ERROR unexpected `...`
    //~^^ ERROR expected one of `:` or `|`, found `)`
        unsafe extern "C" fn f(...) {}
        //~^ ERROR unexpected `...`
        //~^^ ERROR expected one of `:` or `|`, found `)`
    }

    impl A {
        unsafe extern "C" fn f(...) {}
        //~^ ERROR unexpected `...`
        //~^^ ERROR expected one of `:` or `|`, found `)`
    }

    trait A {
        unsafe extern "C" fn f(...) {}
        //[e2018]~^ ERROR unexpected `...`
        //[e2018]~^^ ERROR expected one of `:` or `|`, found `)`
    }

    unsafe extern "C" {
        fn f(...); // no error
    }

    type A = unsafe extern "C" fn(...); // no error
}
