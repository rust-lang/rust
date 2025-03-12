//@ compile-flags: -Znext-solver
#![feature(const_type_id, const_trait_impl)]

use std::any::TypeId;

fn main() {
    const {
        assert!(TypeId::of::<u8>() == TypeId::of::<u8>());
        assert!(TypeId::of::<()>() != TypeId::of::<u8>());
        let _a = TypeId::of::<u8>() < TypeId::of::<u16>();
        //~^ ERROR: cannot call non-const operator in constants
        // can't assert `_a` because it is not deterministic
        // FIXME(const_trait_impl) make it pass
    }
}
