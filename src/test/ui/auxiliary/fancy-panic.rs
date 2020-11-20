#[macro_export]
macro_rules! fancy_panic {
    ($msg:expr) => {
        panic!($msg)
    };
}
