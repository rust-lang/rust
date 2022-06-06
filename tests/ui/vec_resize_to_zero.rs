#![warn(clippy::vec_resize_to_zero)]

fn main() {
    let mut v = vec![1, 2, 3, 4, 5];

    // applicable here
    v.resize(0, 5);

    // not applicable
    v.resize(2, 5);

    let mut v = vec!["foo", "bar", "baz"];

    // applicable here, but only implemented for integer literals for now
    v.resize(0, "bar");

    // not applicable
    v.resize(2, "bar")
}
