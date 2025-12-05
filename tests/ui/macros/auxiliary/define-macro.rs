#[macro_export]
macro_rules! define_macro {
    ($i:ident) => {
        macro_rules! $i { () => {} }
    }
}
