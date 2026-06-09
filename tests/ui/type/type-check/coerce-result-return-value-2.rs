struct A;
struct B;
impl From<A> for B {
    fn from(_: A) -> Self { B }
}
fn foo4(x: Result<(), A>) -> Result<(), B> {
    match true {
        true => x, //~ ERROR mismatched types
        false => x,
    }
}
fn foo5(x: Result<(), A>) -> Result<(), B> {
    match true {
        true => return x, //~ ERROR mismatched types
        false => return x,
    }
}
fn main() {
    let _ = foo4(Ok(()));
    let _ = foo5(Ok(()));
    let _: Result<(), B> = { //~ ERROR mismatched types
        Err(A);
    };
}
