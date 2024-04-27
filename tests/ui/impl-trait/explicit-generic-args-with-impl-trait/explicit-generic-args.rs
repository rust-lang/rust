//@ check-pass

fn foo<T: ?Sized>(_f: impl AsRef<T>) {}

fn main() {
    foo::<str>("".to_string());
}
