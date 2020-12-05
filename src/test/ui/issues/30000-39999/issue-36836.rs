// Previously, in addition to the real cause of the problem as seen below,
// the compiler would tell the user:
//
// ```
// error[E0207]: the type parameter `T` is not constrained by the impl trait, self type, or
// predicates
// ```
//
// With this test, we check that only the relevant error is emitted.

trait Foo {}

impl<T> Foo for Bar<T> {} //~ ERROR cannot find type `Bar` in this scope

fn main() {}
