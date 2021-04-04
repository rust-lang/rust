// ignore-tidy-linelength

struct S;

impl S {
    fn f() {}
}

macro_rules! impl_add {
    ($($n:ident)*) => {
        $(
            fn $n() {
                S::f::<i64>();
                //~^ ERROR this associated function takes 0 type arguments but 1 type argument was supplied
                //~| ERROR this associated function takes 0 type arguments but 1 type argument was supplied
            }
        )*
    }
}

impl_add!(a b);

fn main() { }
