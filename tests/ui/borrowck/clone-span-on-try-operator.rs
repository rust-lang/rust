//@ run-rustfix

#[derive(Clone)]
struct Foo;
impl Foo {
    fn foo(self) {}
}
fn main() {
    let foo = &Foo;
    (*foo).foo(); //~ ERROR cannot move out
}
