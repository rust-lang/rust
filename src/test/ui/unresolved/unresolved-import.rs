// ignore-tidy-linelength

use foo::bar; //~ ERROR unresolved import `foo` [E0432]
              //~^ maybe a missing `extern crate foo;`?

use bar::Baz as x; //~ ERROR unresolved import `bar::Baz` [E0432]
                   //~^ no `Baz` in `bar`. Did you mean to use `Bar`?

use food::baz; //~ ERROR unresolved import `food::baz`
               //~^ no `baz` in `food`. Did you mean to use `bag`?

use food::{beens as Foo}; //~ ERROR unresolved import `food::beens` [E0432]
                          //~^ no `beens` in `food`. Did you mean to use `beans`?

mod bar {
    pub struct Bar;
}

mod food {
    pub use self::zug::baz::{self as bag, Foobar as beans};

    mod zug {
        pub mod baz {
            pub struct Foobar;
        }
    }
}

mod m {
    enum MyEnum {
        MyVariant
    }

    use MyEnum::*; //~ ERROR unresolved import `MyEnum` [E0432]
                   //~^ did you mean `self::MyEnum`?
}

mod items {
    enum Enum {
        Variant
    }

    use Enum::*; //~ ERROR unresolved import `Enum` [E0432]
                 //~^ did you mean `self::Enum`?

    fn item() {}
}

fn main() {}
