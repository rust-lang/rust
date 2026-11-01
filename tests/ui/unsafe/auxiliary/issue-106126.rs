#[macro_export]
macro_rules! foo {
    () => {
        unsafe fn __unsf() {}
        unsafe fn __foo() {
            #[allow(unsafe_op_in_unsafe_fn)] // maximize similarity with original reproducer
            __unsf();
        }
    };
}
