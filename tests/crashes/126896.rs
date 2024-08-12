//@ known-bug: rust-lang/rust#126896
//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes

#![feature(type_alias_impl_trait)]
type Two<'a, 'b> = impl std::fmt::Debug;

fn set(x: &mut isize) -> isize {
    *x
}

fn d(x: Two) {
    let c1 = || set(x);
    c1;
}

fn main() {
}
