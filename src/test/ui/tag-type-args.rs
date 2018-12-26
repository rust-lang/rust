enum Quux<T> { Bar }

fn foo(c: Quux) { assert!((false)); } //~ ERROR wrong number of type arguments

fn main() { panic!(); }
