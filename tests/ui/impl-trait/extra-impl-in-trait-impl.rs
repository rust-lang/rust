//@ run-rustfix

#![allow(dead_code)]
struct S<T>(T);
struct S2;

impl<T: Default> impl Default for S<T> {
    //~^ ERROR: unexpected `impl` keyword
    //~| HELP: remove the extra `impl`
    fn default() -> Self { todo!() }
}

impl impl Default for S2 {
    //~^ ERROR: unexpected `impl` keyword
    //~| HELP: remove the extra `impl`
    fn default() -> Self { todo!() }
}


fn main() {}
