//@ revisions: edi2021 edi2024
//@[edi2024]compile-flags: --edition=2024 -Z unstable-options
//@[edi2021]compile-flags: --edition=2021
// This test ensures that the `_` tok is considered an
// expression on edition 2024.
macro_rules! m2021 {
    ($e:expr_2021) => {
        $e = 1;
    };
}

macro_rules! m2024 {
    ($e:expr) => {
        $e = 1;
    };
}

fn main() {
    m2021!(_); //~ ERROR: no rules expected the token `_`
    m2024!(_); //[edi2021]~ ERROR: no rules expected the token `_`
}
