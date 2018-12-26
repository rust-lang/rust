struct S;

impl S {
    fn f() {}
}

macro_rules! impl_add {
    ($($n:ident)*) => {
        $(
            fn $n() {
                S::f::<i64>();
                //~^ ERROR wrong number of type arguments
            }
        )*
    }
}

impl_add!(a b);

fn main() {}
