#[macro_export]
macro_rules! m {
    () => {
        fn f() {}
        g(); // g is not defined at macro definition site
    }
}
