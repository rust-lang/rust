//@ compile-flags: -Z min-recursion-limit=64
//@ check-pass
#![recursion_limit = "256"]

macro_rules! count {
    () => {};
    ($_:tt $($rest:tt)*) => { count!($($rest)*) };
}

fn main() {
    // 200
    count!(
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
    );
}
