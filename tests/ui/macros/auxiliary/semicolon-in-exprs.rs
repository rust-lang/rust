#[macro_export]
macro_rules! outer {
    ($inner:ident) => { $inner![1, 2, 3]; };
}
