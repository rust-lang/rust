#![feature(generators)]

unsafe fn foo() {
    let mut ga = static || {
        yield 1;
    };
}
