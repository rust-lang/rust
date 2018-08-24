pub fn increment(x: usize) -> usize {
    x + 1
}

#[macro_export]
macro_rules! increment {
    ($x:expr) => ($crate::increment($x))
}

pub fn check_local() {
    assert_eq!(increment!(3), 4);
}
