struct Struct<T>(T);

impl<T> Struct<T> {
    const CONST: fn() = || {
        struct _Obligation where T:; //~ ERROR can't use generic parameters from outer item
    };
}

fn main() {}
