// Ensures that all `fn` forms can have all the function qualifiers syntactically.

// edition:2018

#![feature(const_extern_fn)]
#![feature(const_fn)]

fn main() {
    async fn ff1() {} // OK.
    unsafe fn ff2() {} // OK.
    const fn ff3() {} // OK.
    extern "C" fn ff4() {} // OK.
    const /* async */ unsafe extern "C" fn ff5() {} // OK.
    //^ FIXME(Centril): `async` should be legal syntactically, ensure it's illegal semantically.

    trait X {
        async fn ft1(); //~ ERROR trait fns cannot be declared `async`
        unsafe fn ft2(); // OK.
        const fn ft3(); //~ ERROR trait fns cannot be declared const
        extern "C" fn ft4(); // OK.
        /* const */ async unsafe extern "C" fn ft5();
        //~^ ERROR trait fns cannot be declared `async`
        //^ FIXME(Centril): `const` should be legal syntactically, ensure it's illegal semantically.
    }

    struct Y;
    impl X for Y {
        async fn ft1() {} //~ ERROR trait fns cannot be declared `async`
        //~^ ERROR method `ft1` has an incompatible type for trait
        unsafe fn ft2() {} // OK.
        const fn ft3() {} //~ ERROR trait fns cannot be declared const
        extern "C" fn ft4() {}
        /* const */ async unsafe extern "C" fn ft5() {}
        //~^ ERROR trait fns cannot be declared `async`
        //~| ERROR method `ft5` has an incompatible type for trait
        //^ FIXME(Centril): `const` should be legal syntactically, ensure it's illegal semantically.
    }

    impl Y {
        async fn fi1() {} // OK.
        unsafe fn fi2() {} // OK.
        const fn fi3() {} // OK.
        extern "C" fn fi4() {} // OK.
        /* const */ async unsafe extern "C" fn fi5() {} // OK.
        //^ FIXME(Centril): `const` should be legal syntactically, ensure it's illegal semantically.
    }

    extern {
        async fn fe1(); //~ ERROR functions in `extern` blocks cannot have qualifiers
        unsafe fn fe2(); //~ ERROR functions in `extern` blocks cannot have qualifiers
        const fn fe3(); //~ ERROR functions in `extern` blocks cannot have qualifiers
        extern "C" fn fe4(); //~ ERROR functions in `extern` blocks cannot have qualifiers
        /* const */ async unsafe extern "C" fn fe5();
        //~^ ERROR functions in `extern` blocks cannot have qualifiers
        //^ FIXME(Centril): `const` should be legal syntactically, ensure it's illegal semantically.
    }
}
