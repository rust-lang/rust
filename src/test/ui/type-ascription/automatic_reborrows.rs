#![feature(type_ascription)]

struct Foo<'a> {
  a : &'a [u32], 
}

impl Foo<'_> {
  fn foo_that<'a>(&self, p : &'a [u32]) {}
}  


fn main() {
  let arr = [4,5,6];
  let r = &arr;
  let foo = Foo {a : (r : &[u32]) };
    //~^ ERROR: type ascriptions are not

  let fooer = Foo {a : r};
  fooer.foo_that(r : &[u32]);
    //~^ ERROR: type ascriptions are not
}
