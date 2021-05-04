fn foo<const X: ()>() {} //~ ERROR `()` is forbidden as the type of a const parameter

struct Foo<const X: usize>([(); X]);

fn main() {}
