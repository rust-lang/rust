//@ edition:2015..2021
#[macro_export]
macro_rules! fancy_panic {
    () => {
        panic!("{}");
    };
    ($msg:expr) => {
        panic!($msg)
    };
}
