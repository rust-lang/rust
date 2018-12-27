// run-pass
macro_rules! default {
    ($($x:tt)*) => { $($x)* }
}

default! {
    struct A;
}

impl A {
    default! {
        fn foo(&self) {}
    }
}

fn main() {
    A.foo();
}
