// https://github.com/rust-lang/rust/issues/7784
//@ run-pass

use std::ops::Add;

fn foo<T: Add<Output=T> + Clone>([x, y, z]: [T; 3]) -> (T, T, T) {
    (x.clone(), x.clone() + y.clone(), x + y + z)
}
fn bar(a: &'static str, b: &'static str) -> [&'static str; 4] {
    [a, b, b, a]
}

fn main() {
    assert_eq!(foo([1, 2, 3]), (1, 3, 6));

    let [a, b, c, d] = bar("foo", "bar");
    assert_eq!(a, "foo");
    assert_eq!(b, "bar");
    assert_eq!(c, "bar");
    assert_eq!(d, "foo");

    let [a, _, _, d] = bar("baz", "foo");
    assert_eq!(a, "baz");
    assert_eq!(d, "baz");

    let out = bar("baz", "foo");
    let [a, xs @ .., d] = out;
    assert_eq!(a, "baz");
    assert_eq!(xs, ["foo", "foo"]);
    assert_eq!(d, "baz");
}
