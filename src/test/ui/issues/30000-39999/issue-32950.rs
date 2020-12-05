#![feature(concat_idents)]

#[derive(Debug)]
struct Baz<T>(
    concat_idents!(Foo, Bar) //~ ERROR `derive` cannot be used on items with type macros
                             //~^ ERROR cannot find type `FooBar` in this scope
);

fn main() {}
