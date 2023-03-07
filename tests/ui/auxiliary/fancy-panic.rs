#[macro_export]
macro_rules! fancy_panic {
    () => {
        panic!("{}");
    };
    ($msg:expr) => {
        panic!($msg)
    };
}
