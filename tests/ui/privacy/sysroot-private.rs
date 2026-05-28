//! Test that private dependencies of `std` that live in the sysroot do not reach through to
//! diagnostics.
//!
//! This test would be more robust if we could patch the sysroot with an "evil" crate that
//! provided known types that we control; however, this would effectively require rebuilding
//! `std` (or patching crate metadata). So, this test relies on what is currently public API
//! of `std`'s dependencies, but may not be robust against dependency upgrades/changes.

//@ only-unix Windows sysroots seem to not expose this dependency
//@ ignore-emscripten neither does Emscripten
//@ revisions: default rustc_private_enabled

// Enabling `rustc_private` should `std`'s dependencies accessible, so they should show up
// in diagnostics. NB: not all diagnostics are affected by this.
#![cfg_attr(rustc_private_enabled, feature(rustc_private))]
#![crate_type = "lib"]

trait Trait { type Bar; }

// Attempt to get a suggestion for `gimli::read::op::EvaluationStoreage`, which should not be
// present in diagnostics (it is a dependency of the compiler).
type AssociatedTy = dyn Trait<ExpressionStack = i32, Bar = i32>;
//~^ ERROR associated type `ExpressionStack` not found
//[rustc_private_enabled]~| NOTE there is an associated type `ExpressionStack` in the trait `gimli::read::op::EvaluationStorage`

// Attempt to get a suggestion for `hashbrown::Equivalent`
trait Trait2<K>: Equivalent<K> {}
//~^ ERROR cannot find trait
//~| NOTE not found

// Attempt to get a suggestion for `hashbrown::Equivalent::equivalent`
fn trait_member<T>(val: &T, key: &K) -> bool {
    //~^ ERROR cannot find type `K`
    //~| NOTE similarly named
    val.equivalent(key)
}

// Attempt to get a suggestion for `memchr::memchr2`
fn free_function(buf: &[u8]) -> Option<usize> {
    memchr2(b'a', b'b', buf)
    //~^ ERROR cannot find function
    //~| NOTE not found
}
