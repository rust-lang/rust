//@ check-pass

macro_rules! test_expr {
    ($expr:expr) => {};
}

macro_rules! test_ty {
    ($a:ty | $b:ty) => {};
}

fn main() {
    test_expr!(a as fn() -> B | C);
    // Do not break the `|` operator.

    test_expr!(|_: fn() -> B| C | D);
    // Do not break `-> Ret` in closure args.

    test_ty!(A | B);
    // We can't support anon enums in arbitrary positions.

    test_ty!(fn() -> A | B);
    // Don't break fn ptrs.

    test_ty!(impl Fn() -> A | B);
    // Don't break parenthesized generics.
}
