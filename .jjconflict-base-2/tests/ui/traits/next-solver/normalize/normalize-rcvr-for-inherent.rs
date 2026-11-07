//@ compile-flags: -Znext-solver
//@ check-pass

// Verify that we can assemble inherent impl candidates on a possibly
// unnormalized self type.

trait Foo {
    type Assoc;
}
impl Foo for i32 {
    type Assoc = Bar;
}

struct Bar;
impl Bar {
    fn method(&self) {}
}

fn build<T: Foo>(_: T) -> T::Assoc {
    todo!()
}

fn main() {
    build(1i32).method();
}
