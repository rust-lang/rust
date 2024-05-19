struct Foo(u32);

fn test() -> Foo { Foo } //~ ERROR mismatched types

fn main() {}
