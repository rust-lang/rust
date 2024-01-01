// build-pass
// Tests that paths in `pub(...)` don't fail HIR verification.




pub(self) use self::my_mod::Foo;

mod my_mod {
    pub(super) use self::Foo as Bar;
    pub(in super::my_mod) use self::Foo as Baz;

    pub struct Foo;
}

fn main() {}
