// Regression test for <https://github.com/rust-lang/rust/issues/140785>.

fn main() {
    () += { let x; };
    //~^ ERROR binary assignment operation `+=` cannot be applied to type `()`
    //~| ERROR invalid left-hand side of assignment
}
