#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete

trait Foo {}

pub fn lol(x: dyn* Foo + Send) {
    x as dyn* Foo;
    //~^ ERROR casting `(dyn* Foo + Send + 'static)` as `dyn* Foo` is invalid
}

fn lol2(x: &dyn Foo) {
    *x as dyn* Foo;
    //~^ ERROR `dyn Foo` needs to have the same ABI as a pointer
}

fn main() {}
