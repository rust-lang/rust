//@ check-pass

#![feature(type_alias_impl_trait, rustc_attrs)]
mod foo {
    pub type T = impl Sized;
    // The concrete type referred by impl-trait-type-alias(`T`) is guaranteed
    // to be the same as where it occurs, whereas `impl Trait`'s instance is location sensitive;
    // so difference assertion should not be declared on impl-trait-type-alias's instances.
    // for details, check RFC-2515:
    // https://github.com/rust-lang/rfcs/blob/master/text/2515-type_alias_impl_trait.md

    fn bop(_: T) {
        super::take(|| {});
        super::take(|| {});
    }
}
use foo::*;

fn take(_: fn() -> T) {}

fn main() {}
