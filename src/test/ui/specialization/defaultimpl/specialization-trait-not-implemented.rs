// Tests that:
// - default impls do not have to supply all items and
// - a default impl does not count as an impl (in this case, an incomplete default impl).

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


fn main() {
    println!("{}", MyStruct.foo_one());
    //~^ ERROR no method named `foo_one` found for type `MyStruct` in the current scope
}
