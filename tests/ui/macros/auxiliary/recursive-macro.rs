#[macro_export]
macro_rules! recursive_macro {
    (outer $tt:tt) => { $crate::recursive_macro!(inner $tt) };
    (inner $tt:tt) => { $tt + 2 };
}
