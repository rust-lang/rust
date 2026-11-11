// Test that the NLL solver cannot find a solution
// for `exists<R1> { forall<R1> { R2: R1 } }`.
//
// In this test, the impl should match `fn(T)` for some `T`,
// but we ask it to match `for<'a> fn(&'a ())`. Due to argument
// contravariance, this effectively requires a `T = &'b ()` where
// `forall<'a> { 'a: 'b }`. Therefore, we get an error.
//
// Note the use of `-Zno-leak-check` here. This is presently required in order
// to skip the leak-check errors.
//
// c.f. Issue #57642.
//
//@ compile-flags:-Zno-leak-check

trait Y {
    type F;
    fn make_f() -> Self::F;
}

impl<T> Y for fn(T) {
    type F = fn(T);

    fn make_f() -> Self::F {
        |_| {}
    }
}

fn main() {
    let _x = <fn(&())>::make_f();
    //~^ ERROR implementation of `Y` is not general enough
    //~| ERROR implementation of `Y` is not general enough
    //~| ERROR implementation of `Y` is not general enough
}
