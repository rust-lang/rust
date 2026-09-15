#![feature(fn_delegation)]

reuse foo:: < {
    fn foo() {}
    reuse foo;
    //~^ ERROR: the name `foo` is defined multiple times
  }
  >;

fn main() {}
