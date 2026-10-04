// issue#121613

#![feature(more_qualified_paths)]

struct S<T> {
    a: T
}

struct Foo;

trait A {
    type Assoc<T>;
}

impl A for Foo {
    type Assoc<T> = S<T>;
}

fn f() {}

fn main() {
  <Foo as A>::Assoc::<i32> {
    a: 1
  };
  f(|a, b| a.cmp(b));
  //~^ ERROR: type annotations needed
  //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}
