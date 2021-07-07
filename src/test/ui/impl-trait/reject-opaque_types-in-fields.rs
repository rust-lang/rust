#![feature(min_type_alias_impl_trait)]

type ImplCopy = impl Copy;
//~^ ERROR could not find defining uses

enum Wrapper {
//~^ ERROR type alias impl traits are not allowed as field types in enums
    First(ImplCopy),
    Second
}

type X = impl Iterator<Item = u64> + Unpin;
//~^ ERROR could not find defining uses

struct Foo(X);
//~^ ERROR type alias impl traits are not allowed as field types in structs

impl Foo {
    fn new(z: Vec<u64>) -> Self {
        Foo(z.into_iter())
        //~^ ERROR mismatched types
    }
}

struct Bar {a : X}
//~^ ERROR type alias impl traits are not allowed as field types in structs

impl Bar {
  fn new(z: Vec<u64>) -> Self {
    Bar {a: z.into_iter() }
    //~^ ERROR mismatched types
  }
}

union MyUnion {
  //~^ ERROR type alias impl traits are not allowed as field types in unions
  a: X,
  //~^ ERROR unions may not contain fields that need dropping
}


fn main() {}
