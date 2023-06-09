use std::ops::Deref;
trait Foo {
    type Bar<'a>: Deref<Target = <Self>::Bar<Target = Self>>;
    //~^ ERROR associated type takes 1 lifetime argument but 0 lifetime arguments were supplied
    //~| ERROR associated type bindings are not allowed here
    //~| HELP add missing
}

fn main() {}
