// run-pass
// ignore-pretty pretty-printing is unhygienic

#[macro_export]
macro_rules! m {
    () => {
        fn f() {} // (2)
        g(); // (1)
    }
}
