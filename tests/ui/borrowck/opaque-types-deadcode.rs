//@ compile-flags:-Zverbose-internals

#![feature(rustc_attrs)]
#![rustc_hidden_type_of_opaques]

trait CallMeMaybe<'a, 'b> {
    fn mk() -> Self;
    fn subtype<T>(self, x: &'b T) -> &'a T;
}

struct Foo<'a, 'b: 'a>(&'a (), &'b ());
impl<'a, 'b> CallMeMaybe<'a, 'b> for Foo<'a, 'b> {
    fn mk() -> Self {
        Foo(&(), &())
    }

    fn subtype<T>(self, x: &'b T) -> &'a T {
        x
    }
}

fn good_bye() -> ! {
    panic!();
}

fn foo<'a, 'b: 'a>() -> impl CallMeMaybe<'a, 'b> {
    //~^ ERROR: {type error}
    //~| ERROR: undefined opaque type
    good_bye();
    Foo(&(), &())
}

fn main() {}
