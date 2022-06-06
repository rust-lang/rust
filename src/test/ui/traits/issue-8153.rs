// Test that duplicate methods in impls are not allowed

struct Foo;

trait Bar {
    fn bar(&self) -> isize;
}

impl Bar for Foo {
    fn bar(&self) -> isize {1}
    fn bar(&self) -> isize {2} //~ ERROR duplicate definitions
}

fn main() {
    println!("{}", Foo.bar());
}
