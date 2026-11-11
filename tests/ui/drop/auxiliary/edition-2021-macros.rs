//@ edition:2021

#[macro_export]
macro_rules! edition_2021_block {
    ($($c:tt)*) => {
        { $($c)* }
    }
}
