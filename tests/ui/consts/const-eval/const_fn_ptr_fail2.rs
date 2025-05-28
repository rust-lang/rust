//@ compile-flags: -Zunleash-the-miri-inside-of-you

fn double(x: usize) -> usize {
    x * 2
}
const X: fn(usize) -> usize = double;

const fn bar(x: fn(usize) -> usize, y: usize) -> usize {
    x(y)
    //~^ NOTE inside `bar`
    //~| NOTE the failure occurred here
    //~| NOTE inside `bar`
    //~| NOTE the failure occurred here
}

const Y: usize = bar(X, 2); // FIXME: should fail to typeck someday
//~^ NOTE evaluation of constant value failed
//~| ERROR calling non-const function `double`
const Z: usize = bar(double, 2); // FIXME: should fail to typeck someday
//~^ NOTE evaluation of constant value failed
//~| ERROR calling non-const function `double`

fn main() {
    assert_eq!(Y, 4);
    assert_eq!(Z, 4);
}

//~? WARN skipping const checks
