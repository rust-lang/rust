#[macro_export]
macro_rules! call_it {
    ($f:expr) => {
        (0..100).for_each(&$f)
    };
}
