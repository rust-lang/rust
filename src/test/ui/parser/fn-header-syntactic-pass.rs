// Ensures that all `fn` forms can have all the function qualifiers syntactically.

// check-pass
// edition:2018

#![feature(const_extern_fn)]
//^ FIXME(Centril): move check to ast_validation.

fn main() {}

#[cfg(FALSE)]
fn syntax() {
    async fn f();
    unsafe fn f();
    const fn f();
    extern "C" fn f();
    const /* async */ unsafe extern "C" fn f();
    //^ FIXME(Centril): `async` should be legal syntactically.

    trait X {
        async fn f();
        unsafe fn f();
        const fn f();
        extern "C" fn f();
        /* const */ async unsafe extern "C" fn f();
        //^ FIXME(Centril): `const` should be legal syntactically.
    }

    impl X for Y {
        async fn f();
        unsafe fn f();
        const fn f();
        extern "C" fn f();
        /* const */ async unsafe extern "C" fn f();
        //^ FIXME(Centril): `const` should be legal syntactically.
    }

    impl Y {
        async fn f();
        unsafe fn f();
        const fn f();
        extern "C" fn f();
        /* const */ async unsafe extern "C" fn f();
        //^ FIXME(Centril): `const` should be legal syntactically.
    }

    extern {
        async fn f();
        unsafe fn f();
        const fn f();
        extern "C" fn f();
        /* const */ async unsafe extern "C" fn f();
        //^ FIXME(Centril): `const` should be legal syntactically.
    }
}
