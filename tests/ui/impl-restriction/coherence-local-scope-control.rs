//@ check-pass
//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

#![feature(impl_restriction)]

mod nested {
    pub impl(self) trait Restricted {}

    trait LocalTrait {}

    impl<T: Restricted> LocalTrait for T {}
    impl<T> LocalTrait for &T {}
}

fn main() {}
