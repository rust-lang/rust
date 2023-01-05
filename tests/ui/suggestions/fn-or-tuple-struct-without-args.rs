fn foo(a: usize, b: usize) -> usize { a }

fn bar() -> usize { 42 }

struct S(usize, usize);
enum E {
    A(usize),
    B { a: usize },
}
struct V();

trait T {
    fn baz(x: usize, y: usize) -> usize { x }
    fn bat(x: usize) -> usize { 42 }
    fn bax(x: usize) -> usize { 42 }
    fn bach(x: usize) -> usize;
    fn ban(&self) -> usize { 42 }
    fn bal(&self) -> usize;
}

struct X;

impl T for X {
    fn bach(x: usize) -> usize { 42 }
    fn bal(&self) -> usize { 42 }
}

fn main() {
    let _: usize = foo; //~ ERROR mismatched types
    let _: S = S; //~ ERROR mismatched types
    let _: usize = bar; //~ ERROR mismatched types
    let _: V = V; //~ ERROR mismatched types
    let _: usize = T::baz; //~ ERROR mismatched types
    let _: usize = T::bat; //~ ERROR mismatched types
    let _: E = E::A; //~ ERROR mismatched types
    let _: E = E::B; //~ ERROR expected value, found struct variant `E::B`
    let _: usize = X::baz; //~ ERROR mismatched types
    let _: usize = X::bat; //~ ERROR mismatched types
    let _: usize = X::bax; //~ ERROR mismatched types
    let _: usize = X::bach; //~ ERROR mismatched types
    let _: usize = X::ban; //~ ERROR mismatched types
    let _: usize = X::bal; //~ ERROR mismatched types
    let _: usize = X.ban; //~ ERROR attempted to take value of method
    let _: usize = X.bal; //~ ERROR attempted to take value of method
    let closure = || 42;
    let _: usize = closure; //~ ERROR mismatched types
}
