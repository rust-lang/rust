//@ check-pass
#![allow(incomplete_features)]

fn main() {}

// Tests type equality in a way that avoids coercing `&&T` to `&T`.
trait Eq<T> {}
impl<T> Eq<T> for T {}
fn assert_type_eq<T, U: Eq<T>>(_: T, _: U) {}

#[derive(Copy, Clone)]
struct T;

fn test() {
    let (x,) = &(&T,);
    assert_type_eq(x, &&T);
}
