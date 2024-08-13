// issue#121613

#![feature(more_qualified_paths)]

struct S {}

struct Foo;

trait A {
    type Assoc;
}

impl A for Foo {
    type Assoc = S;
}

fn f() {}

fn main() {
  <Foo as A>::Assoc {};
  f(|a, b| a.cmp(b));
  //~^ ERROR: type annotations needed
  //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}
