#![feature(type_alias_impl_trait)]

mod a {
    type Foo = impl PartialEq<(Foo, i32)>;
    //~^ ERROR: unconstrained opaque type

    struct Bar;

    impl PartialEq<(Bar, i32)> for Bar {
        fn eq(&self, _other: &(Foo, i32)) -> bool {
            //~^ ERROR: `eq` has an incompatible type for trait
            true
        }
    }
}

mod b {
    type Foo = impl PartialEq<(Foo, i32)>;
    //~^ ERROR: unconstrained opaque type

    struct Bar;

    impl PartialEq<(Foo, i32)> for Bar {
        fn eq(&self, _other: &(Bar, i32)) -> bool {
            //~^ ERROR: `eq` has an incompatible type for trait
            true
        }
    }
}

fn main() {}
