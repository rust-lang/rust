#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

trait Hello {}

struct Foo<'a, T>(&'a T);

impl<'a, T> Hello for Foo<'a, &'a T> where Foo<'a, T>: Hello {}

impl Hello for Foo<'static, i32> {}

fn hello<T: Hello>() {}

fn main() {
    hello();
    //~^ ERROR type annotations needed
}
