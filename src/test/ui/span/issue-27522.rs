// Point at correct span for self type

struct SomeType {}

trait Foo {
    fn handler(self: &SomeType); //~ ERROR invalid `self` parameter type
}

fn main() {}
