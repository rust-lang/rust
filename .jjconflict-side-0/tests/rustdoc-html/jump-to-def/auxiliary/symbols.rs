#[macro_export]
macro_rules! symbols {
    ($name:ident = $value:expr) => {
        pub const $name: isize = $value;
    }
}
