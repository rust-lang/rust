use std::ops::Deref;
trait Foo {
    type Bar<'a>: Deref<Target = <Self>::Bar<Target = Self>>;
    //~^ ERROR this associated type takes 1 lifetime argument but 0 lifetime arguments were supplied
    //~| HELP add missing
}

fn main() {}
