#![feature(coroutines)]

unsafe fn foo() {
    let mut ga = #[coroutine]
    static || {
        yield 1;
    };
}
