//@ compile-flags: -Z min-recursion-limit=0
//@ check-pass

// Checks that `min-recursion-limit` cannot lower the default recursion limit

macro_rules! count {
    () => {};
    ($_:tt $($rest:tt)*) => { count!($($rest)*) };
}

fn main() {
    // 100
    count!(
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
        a a a a a a a a a a a a a a a a a a a a
    );
}
