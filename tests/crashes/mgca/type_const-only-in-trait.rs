//@ known-bug: #132980
// Move this test to tests/ui/const-generics/mgca/type_const-only-in-trait.rs
// once fixed.

#![expect(incomplete_features)]
#![feature(associated_const_equality, min_generic_const_args)]

trait GoodTr {
    #[type_const]
    const NUM: usize;
}

struct BadS;

impl GoodTr for BadS {
    const NUM: usize = 42;
}

fn accept_good_tr<const N: usize, T: GoodTr<NUM = { N }>>(_x: &T) {}

fn main() {
    accept_good_tr(&BadS);
}
