#![feature(type_alias_impl_trait)]
//@ check-pass
fn main() {}

type NoReveal = impl std::fmt::Debug;

#[define_opaque(NoReveal)]
fn define_no_reveal() -> NoReveal {
    ""
}

#[define_opaque(NoReveal)]
fn no_reveal(x: NoReveal) {
    let _: &'static str = x;
    let _ = x as &'static str;
}
