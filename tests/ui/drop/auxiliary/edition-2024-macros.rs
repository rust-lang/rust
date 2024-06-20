//@ edition:2024
//@ compile-flags: -Zunstable-options

#[macro_export]
macro_rules! edition_2024_block {
    ($($c:tt)*) => {
        { $($c)* }
    }
}
