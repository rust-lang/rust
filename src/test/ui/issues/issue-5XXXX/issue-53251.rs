struct S;

impl S {
    fn f() {}
}

macro_rules! impl_add {
    ($($n:ident)*) => {
        $(
            fn $n() {
                S::f::<i64>();
                //~^ ERROR this associated function takes 0 generic
                //~| ERROR this associated function takes 0 generic
            }
        )*
    }
}

impl_add!(a b);

fn main() { }
