//@ run-rustfix
struct A;
struct B;
impl From<A> for B {
    fn from(_: A) -> Self { B }
}
fn foo1(x: Result<(), A>) -> Result<(), B> {
    x //~ ERROR mismatched types
}
fn foo2(x: Result<(), A>) -> Result<(), B> {
    return x; //~ ERROR mismatched types
}
fn foo3(x: Result<(), A>) -> Result<(), B> {
    if true {
        x //~ ERROR mismatched types
    } else {
        x //~ ERROR mismatched types
    }
}
fn main() {
    let _ = foo1(Ok(()));
    let _ = foo2(Ok(()));
    let _ = foo3(Ok(()));
}
