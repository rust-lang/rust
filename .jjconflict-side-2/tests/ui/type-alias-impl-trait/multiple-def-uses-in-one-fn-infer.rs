// https://github.com/rust-lang/rust/issues/73481
// This test used to cause unsoundness, since one of the two possible
// resolutions was chosen at random instead of erroring due to conflicts.

#![feature(type_alias_impl_trait)]

type Y<A, B> = impl std::fmt::Debug;

#[define_opaque(Y)]
fn g<A, B>() -> (Y<A, B>, Y<B, A>) {
    //~^ ERROR concrete type differs from previous defining opaque type use
    (42_i64, 60)
}

fn main() {}
