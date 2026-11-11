// We used to lower the ambiguous `T::f(..)` to `<T as B>::f::{type#0}` after emitting the error.
// Meaning we picked one of the candidates and proceeded instead of bailing out early.
// However, sensibly RBV doesn't register any bound vars for ambiguous RTN[^1], so later on when
// wrapping the predicate (here: WellFormed) into a Binder we would correctly fail bound var
// validation (in debug mode).
//
// We now bail out early and thus prevent nonsensical types from getting leaked to subsequent
// compiler passes.
//
// [^1]: It actually maintains its own bespoke lowering function for type-relative paths that
//       relatively closely mirrors the one in HIR ty lowering.

// issue: <https://github.com/rust-lang/rust/issues/139387>
//@ needs-rustc-debug-assertions
#![feature(return_type_notation)]

trait A {
    fn f() -> impl Sized;
}

trait B {
    fn f<'b>() -> impl Sized;
}

fn f<T: A + B>()
where
    T::f(..):, //~ ERROR ambiguous associated function
{
}

fn main() {}
