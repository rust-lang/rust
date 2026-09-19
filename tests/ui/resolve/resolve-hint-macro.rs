//@ run-rustfix
fn main() {
    assert_eq(1, 1);
    //~^ ERROR cannot find function `assert_eq` in this scope
    assert_eq { 1, 1 };
    //~^ ERROR cannot find struct, variant or union type `assert_eq` in this scope
    //~| ERROR expected identifier, found `1`
    //~| ERROR expected identifier, found `1`
    assert[true];
    //~^ ERROR cannot find value `assert` in this scope
}
