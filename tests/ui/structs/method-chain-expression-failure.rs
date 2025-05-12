struct A;
struct B;
struct C;
struct D;
struct E;

impl A {
    fn b(&self) -> B { B }
    fn foo(&self) {}
}

impl B {
    fn c(&self) -> C { C }
}

impl C {
    fn d(&self) -> D { D }
    fn foo(&self) {}
}

impl D {
    fn e(&self) -> E { E }
}

impl E {
    fn f(&self) {}
}
fn main() {
    A.b().c().d().e().foo();
//~^ ERROR no method named `foo` found for struct `E` in the current scope
}
