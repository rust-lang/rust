//! Regression test for <https://github.com/rust-lang/rust/issues/141845>
//! Checks const resolution stability when using inherent associated types
//! and generic const arguments.

//@compile-flags: --crate-type=lib
#![expect(incomplete_features)]
#![feature(inherent_associated_types, min_generic_const_args)]
trait Trait {}

struct Struct<const N: usize>;

type Alias<T: Trait> = Struct<{ Struct::N }>;
//~^ ERROR: missing generics for struct `Struct` [E0107]
