#![crate_type = "lib"]

#[macro_export]
macro_rules! external_decl {
    () => {
        fn external_decl_fn(test: impl ::std::fmt::Display) {
            println!(test);
        }
    };
}
