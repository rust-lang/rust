#![allow(clippy::redundant_clone)] // See #5700

#[derive(PartialEq)]
struct Foo;

struct Bar;

impl std::fmt::Display for Bar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bar")
    }
}

// NOTE: PartialEq<Bar> for T can't be implemented due to the orphan rules
impl<T> PartialEq<T> for Bar
where
    T: AsRef<str> + ?Sized,
{
    fn eq(&self, _: &T) -> bool {
        true
    }
}

// NOTE: PartialEq<Bar> for Foo is not implemented
impl PartialEq<Foo> for Bar {
    fn eq(&self, _: &Foo) -> bool {
        true
    }
}

impl ToOwned for Bar {
    type Owned = Foo;
    fn to_owned(&self) -> Foo {
        Foo
    }
}

impl std::borrow::Borrow<Bar> for Foo {
    fn borrow(&self) -> &Bar {
        static BAR: Bar = Bar;
        &BAR
    }
}

fn main() {
    let b = Bar {};
    if "Hi" == b.to_string() {}

    let f = Foo {};
    if f == b.to_owned() {}
}
