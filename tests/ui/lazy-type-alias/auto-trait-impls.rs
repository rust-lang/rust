//! Auto-trait impls whose self type is a free (lazy) type alias are checked against the
//! type the alias *expands* to, not the alias itself. An alias resolving to an otherwise
//! valid nominal type is accepted, just as if the underlying type had been written
//! directly; an alias resolving to a problematic type is still rejected.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/157756>.

#![feature(lazy_type_alias, auto_traits, negative_impls)]
#![allow(incomplete_features)]

struct Local;

auto trait Marker {}

// Aliases expanding to a local nominal type are accepted for both a cross-crate auto trait
// (`Sync`) and a local one (`Marker`), exactly as if `Local` had been named directly.
type ToLocal = Local;
unsafe impl Sync for ToLocal {}
impl Marker for ToLocal {}

// Nested aliases are expanded transitively, so an alias chain ending in a local nominal type
// is accepted as well.
struct Inner;
type Mid = Inner;
type Nested = Mid;
impl Marker for Nested {}

// A generic alias instantiated with a local nominal type is accepted.
struct Generic;
type Id<T> = T;
impl Marker for Id<Generic> {}

// Negative auto-trait impls go through the same orphan check, so an alias to a local nominal
// type is accepted for them too.
struct Negated;
type ToNeg = Negated;
impl !Marker for ToNeg {}

// An alias expanding to a reference is still rejected for the cross-crate auto trait. Using
// `&'static NotSync` (with `NotSync: !Sync`) avoids overlapping std's `Send for &T` impl, so
// the only error is the auto-trait nominal-type restriction.
struct NotSync;
impl !Sync for NotSync {}
type ToRef = &'static NotSync;
unsafe impl Send for ToRef {}
//~^ ERROR cross-crate traits with a default impl, like `Send`, can only be implemented for a struct/enum type, not `&'static NotSync`

// An alias expanding to a trait object is still rejected for the local auto trait.
type ToDyn = dyn Send;
impl Marker for ToDyn {}
//~^ ERROR traits with a default impl, like `Marker`, cannot be implemented for trait object `(dyn Send + 'static)`

fn main() {}
