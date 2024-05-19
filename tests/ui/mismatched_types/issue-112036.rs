struct Foo;

impl Drop for Foo {
    fn drop(self) {} //~ ERROR method `drop` has an incompatible type for trait
}

fn main() {}
