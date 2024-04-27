trait Foo<T> {
    fn foo(self, f: impl FnOnce());
}

impl<T> Foo<T> for () {
    fn foo(self, f: impl FnOnce()) {
        f()
    }
}

fn main() {
    // FIXME: This should ideally use a fully qualified path
    // without mentioning the generic arguments of `foo`.
    ().foo(|| ()) //~ ERROR type annotations needed
}
