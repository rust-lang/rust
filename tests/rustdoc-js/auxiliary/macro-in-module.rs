#[macro_use]
mod hidden_macro_module {
    #[macro_export]
    macro_rules! vec {
        () => {};
    }
}
