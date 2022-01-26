// check-pass

#![feature(type_alias_impl_trait)]

mod direct {
    type Foo = impl PartialEq<(Foo, i32)>;

    struct Bar;

    impl PartialEq<(Foo, i32)> for Bar {
        fn eq(&self, _other: &(Foo, i32)) -> bool {
            true
        }
    }

    fn foo() -> Foo {
        Bar
    }
}

mod indirect {
    type Foo = impl PartialEq<(Foo, i32)>;

    struct Bar;

    impl PartialEq<(Bar, i32)> for Bar {
        fn eq(&self, _other: &(Bar, i32)) -> bool {
            true
        }
    }

    fn foo() -> Foo {
        Bar
    }
}

fn main() {}
