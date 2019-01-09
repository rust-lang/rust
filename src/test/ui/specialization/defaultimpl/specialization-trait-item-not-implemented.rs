// Tests that default impls do not have to supply all items but regular impls do.

#![feature(specialization)]

trait Foo {
    fn foo_one(&self) -> &'static str;
    fn foo_two(&self) -> &'static str;
}

struct MyStruct;

default impl<T> Foo for T {
    fn foo_one(&self) -> &'static str {
        "generic"
    }
}

impl Foo for MyStruct {}
//~^ ERROR not all trait items implemented, missing: `foo_two` [E0046]

fn main() {
    println!("{}", MyStruct.foo_one());
}
