enum Quux<T> { Bar }
//~^ ERROR: parameter `T` is never used

fn foo(c: Quux) { assert!((false)); } //~ ERROR missing generics for enum `Quux`

fn main() { panic!(); }
