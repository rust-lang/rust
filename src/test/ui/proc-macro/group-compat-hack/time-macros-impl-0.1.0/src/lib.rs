// ignore-test this is not a test

macro_rules! impl_macros {
    ($name:ident) => {
        #[my_macro] struct One($name);
    }
}
