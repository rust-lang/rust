#![feature(type_alias_impl_trait)]

mod a {
    type Foo = impl PartialEq<(Foo, i32)>;

    struct Bar;

    impl PartialEq<(Bar, i32)> for Bar {
        fn eq(&self, _other: &(Foo, i32)) -> bool {
            true
        }
    }
}

mod b {
    type Foo = impl PartialEq<(Foo, i32)>;

    struct Bar;

    impl PartialEq<(Foo, i32)> for Bar {
        //~^ ERROR cannot implement trait on type alias impl trait
        fn eq(&self, _other: &(Bar, i32)) -> bool {
            true
        }
    }
}

fn main() {}
