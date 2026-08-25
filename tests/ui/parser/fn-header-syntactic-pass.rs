// Ensures that all `fn` forms can have all the function qualifiers syntactically.

//@ check-pass
//@ edition:2018

fn main() {}

#[cfg(false)]
fn syntax() {
    async fn f();
    unsafe fn f();
    const fn f();
    extern "C" fn f();
    const async unsafe extern "C" fn f();

    trait X {
        async fn f();
        unsafe fn f();
        const fn f();
        extern "C" fn f();
        const async unsafe extern "C" fn f();
    }

    impl X for Y {
        async fn f();
        unsafe fn f();
        const fn f();
        extern "C" fn f();
        const async unsafe extern "C" fn f();
    }

    impl Y {
        async fn f();
        unsafe fn f();
        const fn f();
        extern "C" fn f();
        const async unsafe extern "C" fn f();
    }

    extern "C" {
        fn f();
        fn f();
        fn f();
        fn f();
        fn f();
    }
}
