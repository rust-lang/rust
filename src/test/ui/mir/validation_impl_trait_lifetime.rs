// check-pass
//! This used to fail MIR validation due to the types on both sides of an assignment not being equal.
fn iter_slice<'a, T>(xs: &'a [T]) -> impl Iterator<Item = &'a T> {
    xs.iter()
}

fn main() {
    iter_slice::<()> as fn(_) -> _;
}
