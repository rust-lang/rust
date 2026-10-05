//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes

// reported as rust-lang/rust#126896. This originally ICE'd
// with polymorphization.

#![feature(type_alias_impl_trait)]
type Two<'a, 'b> = impl std::fmt::Debug;

fn set(x: &mut isize) -> isize {
    *x
}

#[define_opaque(Two)]
fn d(x: Two) {
    let c1 = || set(x); //[current]~ ERROR: expected generic lifetime parameter, found `'_`
    c1;
}

fn main() {}
