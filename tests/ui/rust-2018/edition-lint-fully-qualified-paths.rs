//@ edition: 2015
//@ run-rustfix

#![deny(absolute_paths_not_starting_with_crate)]

mod foo {
    pub(crate) trait Foo {
        type Bar;
    }

    pub(crate) struct Baz {}

    impl Foo for Baz {
        type Bar = ();
    }
}

fn main() {
    let _: <foo::Baz as ::foo::Foo>::Bar = ();
    //~^ ERROR absolute paths must start with
    //~| WARN this is accepted in the current edition
    //~| ERROR absolute paths must start with
    //~| WARN this is accepted in the current edition

    let _: <::foo::Baz as foo::Foo>::Bar = ();
    //~^ ERROR absolute paths must start with
    //~| WARN this is accepted in the current edition
}
