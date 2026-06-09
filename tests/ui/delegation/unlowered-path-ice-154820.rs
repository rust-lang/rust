#![feature(fn_delegation)]

reuse foo:: < { //~ ERROR: failed to resolve delegation callee
  //~^ ERROR: function takes 0 generic arguments but 1 generic argument was supplied
    fn foo() {}
    reuse foo;
    //~^ ERROR: the name `foo` is defined multiple times
  }
  >;

fn main() {}
