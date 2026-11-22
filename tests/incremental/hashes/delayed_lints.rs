// Some lints are emitted in `rustc_attr_parsing`, during ast lowering.
// Emitting these lints is delayed until after ast lowering.
// This test tests that the delayed hints are correctly hashed for incremental.

//@ revisions: cpass1 cpass2 cpass3
//@ compile-flags: -Z query-dep-graph -O -Zincremental-ignore-spans
//@ ignore-backends: gcc
#![feature(rustc_attrs)]

#[doc(alias = "a", alias = "a")]
//~^ WARN doc alias is duplicated

#[cfg_attr(cpass1, doc(alias = "b"))]
#[cfg_attr(not(cpass1), doc(alias = "b", alias = "b"))]
//[cpass2,cpass3]~^ WARN doc alias is duplicated

// The HIR hash should not change between revisions, for this test to be representative
#[rustc_clean(cfg="cpass2")]
#[rustc_clean(cfg="cpass3")]
trait Test {}

fn main() {}
