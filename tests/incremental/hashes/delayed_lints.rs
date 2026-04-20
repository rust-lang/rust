// Some lints are emitted in `rustc_attr_parsing`, during ast lowering.
// Emitting these lints is delayed until after ast lowering.
// This test tests that the delayed hints are correctly hashed for incremental.

//@ check-pass
//@ revisions: cfail1 cfail2 cfail3
//@ compile-flags: -Z query-dep-graph -O -Zincremental-ignore-spans
//@ ignore-backends: gcc
#![feature(rustc_attrs)]

// This attribute is here so the `has_delayed_lints` will be true on all revisions
#[doc(test = 1)]
//~^ WARN `#[doc(test(...)]` takes a list of attributes [invalid_doc_attributes]

// Between revision 1 and 2, the only thing we change is that we add "test = 2"
// This will emit an extra delayed lint, but it will not change the HIR hash.
// We check that even tho the HIR hash didn't change, the extra lint is emitted
#[cfg_attr(cfail1, doc(hidden))]
#[cfg_attr(not(cfail1), doc(hidden, test = 2))]
//[cfail2,cfail3]~^ WARN `#[doc(test(...)]` takes a list of attributes [invalid_doc_attributes]

// The HIR hash should not change between revisions, for this test to be representative
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
trait Test {}

fn main() {}
