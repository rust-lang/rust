// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

// don't reveal the concrete type
type NoReveal = impl std::fmt::Debug;

fn define_no_reveal() -> NoReveal {
    ""
}

fn no_reveal(x: NoReveal) {
    let _: &'static str = x; //~ mismatched types
    let _ = x as &'static str; //~ non-primitive cast
}
