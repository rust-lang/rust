//@ ignore-backends: gcc
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl, const_cmp)]

use std::any::TypeId;

fn main() {
    const {
        assert!(TypeId::of::<u8>() == TypeId::of::<u8>());
        assert!(TypeId::of::<()>() != TypeId::of::<u8>());
        let _a = TypeId::of::<u8>() < TypeId::of::<u16>();
        //~^ ERROR: the trait bound `TypeId: const PartialOrd` is not satisfied
        // FIXME(const_trait_impl) make it pass; requires const comparison of pointers (#53020)
    }
}
