pub fn f() {}

#[macro_export]
macro_rules! m { () => {
    use $crate;
    import_crate_var::f();
} }
