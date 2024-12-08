// Inspired by https://github.com/rust-lang/cargo/issues/14775

pub fn my_lib_func() {}

#[macro_export]
macro_rules! my_lib_macro {
    () => {
        #[cfg(my_lib_cfg)]
        $crate::my_lib_func()
    };
}
