// We don't need those errors. Ideally we would silence them, but to do so we need to move the
// lint from being an early-lint during parsing to a late-lint, because it needs to be aware of
// the types involved.
#![allow(bare_trait_objects)]

struct Foo;

fn foo(_x: Box<Foo + Send>) { } //~ ERROR expected trait, found struct `Foo`

type TypeAlias<T> = Box<dyn Vec<T>>; //~ ERROR expected trait, found struct `Vec`

struct A;
fn a() -> A + 'static { //~ ERROR expected trait, found
    A
}
fn b<'a,T,E>(iter: Iterator<Item=Result<T,E> + 'a>) { //~ ERROR expected trait, found
    panic!()
}
fn c() -> 'static + A { //~ ERROR expected trait, found
    A
}
fn d<'a,T,E>(iter: Iterator<Item='a + Result<T,E>>) { //~ ERROR expected trait, found
    panic!()
}
fn e() -> 'static + A + 'static { //~ ERROR expected trait, found
//~^ ERROR only a single explicit lifetime bound is permitted
    A
}
fn f<'a,T,E>(iter: Iterator<Item='a + Result<T,E> + 'a>) { //~ ERROR expected trait, found
//~^ ERROR only a single explicit lifetime bound is permitted
    panic!()
}
struct Traitor;
trait Trait {}
fn g() -> Traitor + 'static { //~ ERROR expected trait, found struct `Traitor`
    A
}
fn main() {}
