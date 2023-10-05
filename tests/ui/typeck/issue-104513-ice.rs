struct S;
fn f() {
    let _: S<impl Oops> = S; //~ ERROR cannot find trait `Oops` in this scope
    //~^ ERROR `impl Trait` only allowed in function and inherent method argument and return types
}
fn main() {}
