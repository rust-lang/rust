fn foo<const X: ()>() {} //~ ERROR const generics are unstable
//~^ const generics in any position are currently unsupported

struct Foo<const X: usize>([(); X]); //~ ERROR const generics are unstable
//~^ const generics in any position are currently unsupported

fn main() {}
