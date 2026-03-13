//@ compile-flags: -Z min-recursion-limit=256
//@ check-pass
#![recursion_limit = "128"]

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
