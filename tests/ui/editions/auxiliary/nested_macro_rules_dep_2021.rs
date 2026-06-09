//@ edition: 2021

#[macro_export]
macro_rules! make_macro_with_input {
    ($i:ident) => {
        macro_rules! macro_inner_input {
            () => {
                pub fn $i() {}
            };
        }
    };
}

#[macro_export]
macro_rules! make_macro {
    () => {
        macro_rules! macro_inner {
            () => {
                pub fn gen() {}
            };
        }
    };
}
