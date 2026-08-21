//@ check-pass
//
// Unifying two unknown vids can make a two-var goal progress
// (`T: Mirror<T>`). Fulfillment must still walk that goal; it
// must not treat unknown-unknown equate as a skip.

trait Mirror<T> {}
impl<T> Mirror<T> for T {}

fn assert_mirror<T: Mirror<U>, U>(_: T, _: U) {}

fn unify<T>(x: T, y: T) {}

pub fn check() {
    let a = None;
    let b = None;
    assert_mirror(a, b);
    unify(a, b);
}

fn main() {
    check();
}
