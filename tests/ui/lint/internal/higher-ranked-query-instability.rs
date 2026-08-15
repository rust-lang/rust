//@ check-pass
//@ compile-flags: -Zunstable-options

// Make sure we don't try to resolve instances for trait refs that have escaping
// bound vars when computing the query instability lint.

fn foo<T>() where for<'a> &'a [T]: IntoIterator<Item = &'a T> {}

fn main() {
    foo::<()>();
}
