// run-rustfix
fn main() {
    assert_eq!(1, 1);
    //~^ ERROR expected function, found macro `assert_eq`
    assert_eq! { 1, 1 };
    //~^ ERROR expected struct, variant or union type, found macro `assert_eq`
    //~| ERROR expected identifier, found `1`
    //~| ERROR expected identifier, found `1`
    assert![true];
    //~^ ERROR expected value, found macro `assert`
}
