#[macro_export]
macro_rules! bar {
    () => {use std::string::ToString;}
}

#[macro_export]
macro_rules! baz {
    ($i:item) => ($i)
}

#[macro_export]
macro_rules! baz2 {
    ($($i:tt)*) => ($($i)*)
}
