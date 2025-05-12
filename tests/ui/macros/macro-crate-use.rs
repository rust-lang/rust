//@ run-pass

pub fn increment(x: usize) -> usize {
    x + 1
}

#[macro_export]
macro_rules! increment {
    ($x:expr) => ({
        use $crate::increment;
        increment($x)
    })
}

fn main() {
    assert_eq!(increment!(3), 4);
}
