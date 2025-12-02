// https://github.com/rust-lang/rust/issues/87199
// Regression test for issue #87199, where attempting to relax a bound
// other than the only supported `?Sized` would still cause the compiler
// to assume that the `Sized` bound was relaxed.

//@ check-fail

// Check that these function definitions only emit warnings, not errors
fn arg<T: ?Send>(_: T) {}
//~^ ERROR: bound modifier `?` can only be applied to `Sized`
fn ref_arg<T: ?Send>(_: &T) {}
//~^ ERROR: bound modifier `?` can only be applied to `Sized`
fn ret() -> impl Iterator<Item = ()> + ?Send { std::iter::empty() }
//~^ ERROR: bound modifier `?` can only be applied to `Sized`
//~| ERROR: bound modifier `?` can only be applied to `Sized`

// Check that there's no `?Sized` relaxation!
fn main() {
    ref_arg::<i32>(&5);
    ref_arg::<[i32]>(&[5]);
    //~^ ERROR the size for values of type `[i32]` cannot be known
}
