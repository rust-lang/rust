struct S;
fn f() {
    let _: S<impl Oops> = S; //~ ERROR cannot find trait `Oops` in this scope
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
}
fn main() {}
