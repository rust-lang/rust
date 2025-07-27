mod foo {
  struct Self;
  //~^ ERROR expected identifier, found keyword `Self`
}

struct Bar<'Self>;
//~^ ERROR lifetimes cannot use keyword names
//~| ERROR parameter `'Self` is never used

struct Foo;

pub fn main() {
    match 15 {
        ref Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        mut Self => (),
        //~^ ERROR `mut` must be followed by a named binding
        //~| ERROR cannot find unit struct, unit variant or constant `Self`
        ref mut Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        Foo { Self } => (),
        //~^ ERROR expected identifier, found keyword `Self`
        //~| ERROR mismatched types
        //~| ERROR `Foo` does not have a field named `Self`
    }
}

mod m1 {
    extern crate core as Self;
    //~^ ERROR expected identifier, found keyword `Self`
}

mod m2 {
    use std::option::Option as Self;
    //~^ ERROR expected identifier, found keyword `Self`
}

mod m3 {
    trait Self {}
    //~^ ERROR expected identifier, found keyword `Self`
}
