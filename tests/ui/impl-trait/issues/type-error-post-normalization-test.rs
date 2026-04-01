//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes

// This previously introduced a `{type_error}`` in the MIR body
// during the `PostAnalysisNormalize` pass. While the underlying issue
// #135528 did not get fixed, this reproducer no longer ICEs.

#![feature(type_alias_impl_trait)]
type Tait = impl Copy;

fn set(x: &isize) -> isize {
    *x
}

#[define_opaque(Tait)]
fn d(x: Tait) {
    set(x);
}

#[define_opaque(Tait)]
fn other_define() -> Tait {
    () //~^ ERROR concrete type differs from previous defining opaque type use
}

fn main() {}
