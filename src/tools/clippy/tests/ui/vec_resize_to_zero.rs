#![warn(clippy::vec_resize_to_zero)]

fn main() {
    // applicable here
    vec![1, 2, 3, 4, 5].resize(0, 5);

    // not applicable
    vec![1, 2, 3, 4, 5].resize(2, 5);

    // applicable here, but only implemented for integer literals for now
    vec!["foo", "bar", "baz"].resize(0, "bar");

    // not applicable
    vec!["foo", "bar", "baz"].resize(2, "bar")
}
