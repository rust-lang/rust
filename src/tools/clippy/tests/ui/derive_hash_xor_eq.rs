#[derive(PartialEq, Hash)]
struct Foo;

impl PartialEq<u64> for Foo {
    fn eq(&self, _: &u64) -> bool {
        true
    }
}

#[derive(Hash)]
struct Bar;

impl PartialEq for Bar {
    fn eq(&self, _: &Bar) -> bool {
        true
    }
}

#[derive(Hash)]
struct Baz;

impl PartialEq<Baz> for Baz {
    fn eq(&self, _: &Baz) -> bool {
        true
    }
}

#[derive(PartialEq)]
struct Bah;

impl std::hash::Hash for Bah {
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {}
}

#[derive(PartialEq)]
struct Foo2;

trait Hash {}

// We don't want to lint on user-defined traits called `Hash`
impl Hash for Foo2 {}

mod use_hash {
    use std::hash::{Hash, Hasher};

    #[derive(PartialEq)]
    struct Foo3;

    impl Hash for Foo3 {
        fn hash<H: std::hash::Hasher>(&self, _: &mut H) {}
    }
}

fn main() {}
