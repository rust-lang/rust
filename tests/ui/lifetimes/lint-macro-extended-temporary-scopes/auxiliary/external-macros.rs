//! The macros that are in `../user-defined-macros.rs`, but external to test diagnostics.
//@ edition: 2024

#[macro_export]
macro_rules! wrap {
    ($arg:expr) => { { &$arg } }
}

#[macro_export]
macro_rules! print_with_internal_wrap {
    () => { println!("{:?}{}", (), $crate::wrap!(String::new())) }
}
