//@ run-rustfix

struct S;
trait Trait {
    fn foo() {}
}
impl Trait for &S {}
impl Trait for &mut S {}
fn main() {
    let _ = &str::from("value");
    //~^ ERROR the trait bound `str: From<_>` is not satisfied
    //~| ERROR the size for values of type `str` cannot be known at compilation time
    let _ = &mut S::foo();
    //~^ ERROR the trait bound `S: Trait` is not satisfied
    let _ = &S::foo();
    //~^ ERROR the trait bound `S: Trait` is not satisfied
}
