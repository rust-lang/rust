//! Checks trailing commas are accepted in various places:
//! - Generic parameters in function and struct definitions.
//! - Function and method arguments.
//! - Tuple and array literal expressions.
//! - Tuple and array destructuring patterns, including those with `..`.
//! - Enum variant declarations.
//! - Attributes.

//@ run-pass

fn f<T,>(_: T,) {}

struct Foo<T,>(#[allow(dead_code)] T);

struct Bar;

impl Bar {
    fn f(_: isize,) {}
    fn g(self, _: isize,) {}
    fn h(self,) {}
}

enum Baz {
    Qux(#[allow(dead_code)] isize,),
}

#[allow(unused,)]
pub fn main() {
    f::<isize,>(0,);
    let (_, _,) = (1, 1,);
    let [_, _,] = [1, 1,];
    let [_, _, .., _,] = [1, 1, 1, 1,];
    let [_, _, _, ..,] = [1, 1, 1, 1,];

    let x: Foo<isize,> = Foo::<isize,>(1);

    Bar::f(0,);
    Bar.g(0,);
    Bar.h();

    let x = Baz::Qux(1,);
}
