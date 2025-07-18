//! Test that ?Trait bounds are forbidden in supertraits and trait object types.
//!
//! While `?Sized` and other maybe bounds are allowed in type parameter bounds and where clauses,
//! they are explicitly forbidden in certain syntactic positions:
//! - As supertraits in trait definitions
//! - In trait object type expressions
//!
//! See https://github.com/rust-lang/rust/issues/20503

trait Tr: ?Sized {}
//~^ ERROR `?Trait` is not permitted in supertraits

type A1 = dyn Tr + (?Sized);
//~^ ERROR `?Trait` is not permitted in trait object types
type A2 = dyn for<'a> Tr + (?Sized);
//~^ ERROR `?Trait` is not permitted in trait object types

fn main() {}
