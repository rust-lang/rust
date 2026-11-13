//! Regression test for https://github.com/rust-lang/rust/issues/53251
struct S;

impl S {
    fn f() {}
}

macro_rules! impl_add {
    ($($n:ident)*) => {
        $(
            fn $n() {
                S::f::<i64>();
                //~^ ERROR associated function takes 0 generic
                //~| ERROR associated function takes 0 generic
            }
        )*
    }
}

impl_add!(a b);

fn main() { }
