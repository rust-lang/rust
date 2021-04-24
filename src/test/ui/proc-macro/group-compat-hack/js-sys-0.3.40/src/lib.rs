// ignore-test this is not a test

macro_rules! arrays {
    ($name:ident) => {
        #[my_macro] struct Two($name);
    }
}
