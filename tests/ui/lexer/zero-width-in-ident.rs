//@ check-pass
//! As of Unicode 15.1, zero-width-non-joiners and zero-width-joiners
//! are included in `XID_Continue`. Test that they are accepted in identifiers,
//! and an `uncommon_codepoints` lint is emitted.
#![expect(uncommon_codepoints)]

// This name contains ZWNJ (U+200C)
fn zero‌nonjoiner() {}
// This name contains ZWJ (U+200D)
fn zero‍joiner() {}

fn main() {}
