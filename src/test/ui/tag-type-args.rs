enum Quux<T> { Bar }

fn foo(c: Quux) { assert!((false)); } //~ ERROR missing generics for enum `Quux`

fn main() { panic!(); }
