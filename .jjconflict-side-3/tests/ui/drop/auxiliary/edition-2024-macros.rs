//@ edition:2024

#[macro_export]
macro_rules! edition_2024_block {
    ($($c:tt)*) => {
        { $($c)* }
    }
}
