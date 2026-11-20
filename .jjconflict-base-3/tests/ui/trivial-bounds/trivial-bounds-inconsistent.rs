//@ run-pass

// Check that tautalogically false bounds are accepted, and are used
// in type inference.
#![feature(trivial_bounds)]
#![allow(unused)]

pub trait Foo {
    fn test(&self);
}

fn generic_function<X: Foo>(x: X) {}

enum E where i32: Foo { V } //~ WARNING trivial_bounds

struct S where i32: Foo; //~ WARNING trivial_bounds

trait T where i32: Foo {} //~ WARNING trivial_bounds

union U where i32: Foo { f: i32 } //~ WARNING trivial_bounds

type Y where i32: Foo = ();
//~^ WARNING type_alias_bounds
//~| WARNING trivial_bounds

impl Foo for () where i32: Foo { //~ WARNING trivial_bounds
    fn test(&self) {
        3i32.test();
        Foo::test(&4i32);
        generic_function(5i32);
    }
}

fn f() where i32: Foo { //~ WARNING trivial_bounds
    let s = S;
    3i32.test();
    Foo::test(&4i32);
    generic_function(5i32);
}

fn g() where &'static str: Foo { //~ WARNING trivial_bounds
    "Foo".test();
    Foo::test(&"Foo");
    generic_function("Foo");
}

trait A {}

impl A for i32 {}

struct Dst<X: ?Sized> {
    x: X,
}

struct TwoStrs(str, str) where str: Sized; //~ WARNING trivial_bounds

fn unsized_local() where for<'a> Dst<dyn A + 'a>: Sized { //~ WARNING trivial_bounds
    let x: Dst<dyn A> = *(Box::new(Dst { x: 1 }) as Box<Dst<dyn A>>);
}

fn return_str() -> str where str: Sized { //~ WARNING trivial_bounds
    *"Sized".to_string().into_boxed_str()
}

fn use_op(s: String) -> String where String: ::std::ops::Neg<Output=String> {
    //~^ WARNING trivial_bounds
    -s
}

fn use_for() where i32: Iterator { //~ WARNING trivial_bounds
    for _ in 2i32 {}
}

fn main() {}
