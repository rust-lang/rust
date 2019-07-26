// run-pass

// Tests that we can combine a default impl that supplies one method with a
// full impl that supplies the other, and they can invoke one another.

#![feature(specialization)]

trait Foo {
    fn foo_one(&self) -> &'static str;
    fn foo_two(&self) -> &'static str;
    fn foo_three(&self) -> &'static str;
}

struct MyStruct;

default impl<T> Foo for T {
    fn foo_one(&self) -> &'static str {
        self.foo_three()
    }
}

impl Foo for MyStruct {
    fn foo_two(&self) -> &'static str {
        self.foo_one()
    }

    fn foo_three(&self) -> &'static str {
        "generic"
    }
}

fn main() {
    assert!(MyStruct.foo_two() == "generic");
}
