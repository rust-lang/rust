//@ check-pass


trait Foo {
    fn early<'a, T: 'a>(x: &'a T) -> impl Iterator<Item = impl Into<&'a T>>;

    fn late<'a, T>(x: &'a T) -> impl Iterator<Item = impl Into<&'a T>>;
}

fn main() {}
