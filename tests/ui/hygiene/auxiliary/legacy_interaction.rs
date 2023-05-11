#[macro_export]
macro_rules! m {
    () => {
        fn f() {} // (2)
        g(); // (1)
    }
}
