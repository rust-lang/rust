trait Trait {
    fn foo() -> Result<(), ()>;
}

fn some_f() -> Result<(), ()> {
    Ok(())
}

macro_rules! impl_from_endian_bytes {
    ($($t:ty),*$(,)?) => {$(
        impl Trait for $t {
            fn foo() -> Result<(), ()> {
                Ok(some_f()?)
            }
        }
    )*};
}

struct S1;
struct S2;
struct S3;
struct S4;
struct S5;
struct S6;
struct S7;
struct S8;
struct S9;

impl_from_endian_bytes!(S1, S2, S3, S4, S5, S6, S7, S8, S9);
