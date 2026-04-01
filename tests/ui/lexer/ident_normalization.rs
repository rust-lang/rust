//@check-pass
//@edition:2021

#![allow(non_snake_case)]

// Tests that identifiers are NFC-normalized as per
// https://rust-lang.github.io/rfcs/2457-non-ascii-idents.html

// Note that in the first argument of each function `K` is LATIN CAPITAL LETTER K
// and in the second it is K (KELVIN SIGN).

fn ident_nfc<K>(_p1: K, _p2: K) {}

fn raw_ident_nfc<K>(_p1: r#K, _p2: r#K) {}

fn lifetime_nfc<'K>(_p1: &'K str, _p2: &'K str) {}

fn raw_lifetime_nfc<'K>(_p1: &'r#K str, _p2: &'r#K str) {}

fn main() {}
