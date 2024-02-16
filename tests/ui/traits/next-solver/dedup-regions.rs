//@ compile-flags: -Znext-solver
//@ check-pass

struct A(*mut ());

unsafe impl Send for A where A: 'static {}

macro_rules! mk {
    ($name:ident $ty:ty) => {
        struct $name($ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty);
    };
}

mk!(B A);
mk!(C B);
mk!(D C);
mk!(E D);
mk!(F E);
mk!(G F);
mk!(H G);
mk!(I H);
mk!(J I);
mk!(K J);
mk!(L K);
mk!(M L);

fn needs_send<T: Send>() {}

fn main() {
    needs_send::<M>();
}
