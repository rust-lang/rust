// Point at correct span for self type

struct SomeType {}

trait Foo {
    fn handler(self: &SomeType); //~ ERROR invalid method receiver type
}

fn main() {}
