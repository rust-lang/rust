//@ revisions: edi2021 edi2024
//@[edi2024]compile-flags: --edition=2024 -Z unstable-options
//@[edi2021]compile-flags: --edition=2021

// This test ensures that the inline const match only on edition 2024
#![feature(expr_fragment_specifier_2024)]
#![allow(incomplete_features)]

macro_rules! m2021 {
    ($e:expr_2021) => {
        $e
    };
}

macro_rules! m2024 {
    ($e:expr) => {
        $e
    };
}
fn main() {
    m2021!(const { 1 }); //~ ERROR: no rules expected the token `const`
    m2024!(const { 1 }); //[edi2021]~ ERROR: no rules expected the token `const`
}
