fn foo<const X: ()>() {} //~ ERROR `()` is forbidden as the type of a const generic parameter

struct Foo<const X: usize>([(); X]);

fn main() {}
