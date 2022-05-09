fn foo<T: ?Sized>(_f: impl AsRef<T>) {}

fn main() {
    foo::<str, String>("".to_string()); //~ ERROR E0107
}
