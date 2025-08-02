// FIXME(is): This syntax is a placeholder.
#[macro_export]
macro_rules! is {
    ($($args:tt)*) => {
        builtin # is($($args)*)
    }
}
