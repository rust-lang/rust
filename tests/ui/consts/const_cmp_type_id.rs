//@ compile-flags: -Znext-solver
#![feature(const_type_id, const_trait_impl)]

use std::any::TypeId;

fn main() {
    const {
        assert!(TypeId::of::<u8>() == TypeId::of::<u8>());
        //~^ ERROR the trait bound `TypeId: const PartialEq` is not satisfied
        assert!(TypeId::of::<()>() != TypeId::of::<u8>());
        //~^ ERROR the trait bound `TypeId: const PartialEq` is not satisfied
        let _a = TypeId::of::<u8>() < TypeId::of::<u16>();
        // can't assert `_a` because it is not deterministic
        // FIXME(const_trait_impl) make it pass
    }
}
