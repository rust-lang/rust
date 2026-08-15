//@ compile-flags: -Z min-recursion-limit=256
//@ check-pass

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
