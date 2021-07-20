// Regression test for issue #87199, where attempting to relax a bound
// other than the only supported `?Sized` would still cause the compiler
// to assume that the `Sized` bound was relaxed.

// check-fail

// Check that these function definitions only emit warnings, not errors
fn arg<T: ?Send>(_: T) {}
//~^ warning: default bound relaxed for a type parameter, but this does nothing
fn ref_arg<T: ?Send>(_: &T) {}
//~^ warning: default bound relaxed for a type parameter, but this does nothing
fn ret() -> impl Iterator<Item = ()> + ?Send { std::iter::empty() }
//~^ warning: default bound relaxed for a type parameter, but this does nothing

// Check that there's no `?Sized` relaxation!
fn main() {
    ref_arg::<i32>(&5);
    ref_arg::<[i32]>(&[5]);
    //~^ the size for values of type `[i32]` cannot be known
}
