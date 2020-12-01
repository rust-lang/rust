// ignore-test this is not a test

macro_rules! tuple_from_req {
    ($T:ident) => {
        #[my_macro] struct Four($T);
    }
}
