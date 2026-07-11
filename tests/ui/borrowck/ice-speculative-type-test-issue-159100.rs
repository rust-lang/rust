//@ check-pass
//@ edition: 2021

#![feature(generic_const_exprs)]
//~^ WARNING the feature `generic_const_exprs` is incomplete

fn main() {
    async { std::any::TypeId::of::<[u8; 1 << 3]> };
}
