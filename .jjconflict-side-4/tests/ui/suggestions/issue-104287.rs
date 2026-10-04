// The purpose of this test is not to validate the output of the compiler.
// Instead, it ensures the suggestion is generated without performing an arithmetic overflow.

struct S;
impl S {
    fn foo(&self) {}
}
fn main() {
    let x = S;
    foo::<()>(x);
    //~^ ERROR method takes 0 generic arguments but 1 generic argument was supplied
    //~| ERROR cannot find function `foo` in this scope
}
