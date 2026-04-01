#![allow(clippy::unnecessary_operation)]
#![allow(clippy::implicit_clone)]

fn main() {
    let x = &Baz;
    let y = &Baz;
    y.to_owned() == *x;
    //~^ cmp_owned

    let x = &&Baz;
    let y = &Baz;
    y.to_owned() == **x;
    //~^ cmp_owned

    let x = 0u32;
    let y = U32Wrapper(x);
    let _ = U32Wrapper::from(x) == y;
}

struct Foo;

impl PartialEq for Foo {
    fn eq(&self, other: &Self) -> bool {
        self.to_owned() == *other
        //~^ cmp_owned
    }
}

impl ToOwned for Foo {
    type Owned = Bar;
    fn to_owned(&self) -> Bar {
        Bar
    }
}

#[derive(PartialEq, Eq)]
struct Baz;

impl ToOwned for Baz {
    type Owned = Baz;
    fn to_owned(&self) -> Baz {
        Baz
    }
}

#[derive(PartialEq, Eq)]
struct Bar;

impl PartialEq<Foo> for Bar {
    fn eq(&self, _: &Foo) -> bool {
        true
    }
}

impl std::borrow::Borrow<Foo> for Bar {
    fn borrow(&self) -> &Foo {
        static FOO: Foo = Foo;
        &FOO
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct U32Wrapper(u32);
impl From<u32> for U32Wrapper {
    fn from(x: u32) -> Self {
        Self(x)
    }
}
impl PartialEq<u32> for U32Wrapper {
    fn eq(&self, other: &u32) -> bool {
        self.0 == *other
    }
}
impl PartialEq<U32Wrapper> for u32 {
    fn eq(&self, other: &U32Wrapper) -> bool {
        *self == other.0
    }
}
