#![feature(generic_associated_types)]

trait Foo {
    type F<'a>;

    fn identity<'a>(t: &'a Self::F<'a>) -> &'a Self::F<'a> { t }
}

impl <T, T1> Foo for T {
    type F<T1> = &[u8];
      //~^ ERROR: the name `T1` is already used for
      //~| ERROR: missing lifetime specifier
}

fn main() {}
