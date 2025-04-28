use foo::bar;
//~^ ERROR unresolved import `foo` [E0432]
//~| NOTE use of unresolved module or unlinked crate `foo`
//~| HELP you might be missing a crate named `foo`
//~| SUGGESTION extern crate foo;

use bar::Baz as x;
//~^ ERROR unresolved import `bar::Baz` [E0432]
//~| NOTE no `Baz` in `bar`
//~| HELP a similar name exists in the module
//~| SUGGESTION Bar

use food::baz;
//~^ ERROR unresolved import `food::baz`
//~| NOTE no `baz` in `food`
//~| HELP a similar name exists in the module
//~| SUGGESTION bag

use food::{beens as Foo};
//~^ ERROR unresolved import `food::beens` [E0432]
//~| NOTE no `beens` in `food`
//~| HELP a similar name exists in the module
//~| SUGGESTION beans

mod bar {
    pub struct Bar;
}

mod food {
    pub use self::zug::baz::{self as bag, Foobar as beans};

    mod zug {
        pub mod baz {
        //~^ NOTE module `food::zug::baz` exists but is inaccessible
        //~| NOTE not accessible
            pub struct Foobar;
        }
    }
}

mod m {
    enum MyEnum {
        MyVariant
    }

    use MyEnum::*;
    //~^ ERROR unresolved import `MyEnum` [E0432]
    //~| HELP a similar path exists
    //~| SUGGESTION self::MyEnum
}

mod items {
    enum Enum {
        Variant
    }

    use Enum::*;
    //~^ ERROR unresolved import `Enum` [E0432]
    //~| HELP a similar path exists
    //~| SUGGESTION self::Enum

    fn item() {}
}

fn main() {}
