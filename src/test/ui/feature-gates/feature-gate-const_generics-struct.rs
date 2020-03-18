struct Foo<const X: usize>([(); X]); //~ ERROR const generics are unstable

fn main() {}
