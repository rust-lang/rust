//@ normalize-stderr: "\[u8; [0-9]+\]" -> "[u8; N]"
//! Test for https://github.com/rust-lang/rust/pull/152003

#![feature(type_info)]

use std::any::TypeId;

trait Trait {}
impl Trait for [u8; usize::MAX] {}

fn main() {}

const _: () = const {
    TypeId::of::<[u8; usize::MAX]>().trait_info_of_trait_type_id(TypeId::of::<dyn Trait>());
    //~^ ERROR values of the type `[u8; usize::MAX]` are too big for the target architecture
};
const _: () = const {
    TypeId::of::<[u8; usize::MAX]>().trait_info_of::<dyn Trait>();
    //~^ ERROR values of the type `[u8; usize::MAX]` are too big for the target architecture
};
