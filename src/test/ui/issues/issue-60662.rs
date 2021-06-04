// check-pass
// compile-flags: -Z unpretty=hir

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait Animal {
}

fn main() {
    pub type ServeFut = impl Animal;
}
