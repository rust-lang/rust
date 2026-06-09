//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes

// reported as rust-lang/rust#126896

#![feature(type_alias_impl_trait)]
type Two<'a, 'b> = impl std::fmt::Debug;

fn set(x: &mut isize) -> isize {
    *x
}

#[define_opaque(Two)]
fn d(x: Two) {
    let c1 = || set(x); //~ ERROR: expected generic lifetime parameter, found `'_`
    c1;
}

fn main() {}
