// Test that `Box` cannot be used with a lifetime argument.

struct Foo<'a> {
    x: Box<'a, isize> //~ ERROR wrong number of lifetime arguments
}

pub fn main() {
}
