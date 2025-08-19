//! Regression test for <https://github.com/rust-lang/rust/issues/143413>
//! The anonymous lifetime in `c(&())` is desugared by the resolver as an extra lifetime parameter
//! at the end of the `for` binder. Verify that lowering creates the definition for that extra
//! lifetime parameter before lowering `c(&())`.

trait D {}

type A = dyn for<const B: c(&())> D;
//~^ ERROR cannot find type `c` in this scope
//~| ERROR only lifetime parameters can be used in this context

fn main() {}
