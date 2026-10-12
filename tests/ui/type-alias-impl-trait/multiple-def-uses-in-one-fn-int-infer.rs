//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
// https://github.com/rust-lang/rust/issues/73481

// This test previously compiled due to an unsoundness as `Y<B, A>`
// was incorrectly inferred to be `i32`. The new solver now treats
// `Y<B, A>` as a defining use and uses it to constrain `Y<B, A>` to
// `i64` as well.

#![feature(type_alias_impl_trait)]

type Y<A, B> = impl std::fmt::Debug;

#[define_opaque(Y)]
fn g<A, B>() -> (Y<A, B>, Y<B, A>) {
    //[current]~^ ERROR concrete type differs from previous defining opaque type use
    (42_i64, 60)
}

fn main() {}
