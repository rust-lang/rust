//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_type_id, const_trait_impl, effects)]
#![allow(incomplete_features)]

use std::any::TypeId;

fn main() {
    const {
        // FIXME(effects) this isn't supposed to pass (right now) but it did.
        // revisit binops typeck please.
        assert!(TypeId::of::<u8>() == TypeId::of::<u8>());
        assert!(TypeId::of::<()>() != TypeId::of::<u8>());
        let _a = TypeId::of::<u8>() < TypeId::of::<u16>();
        // can't assert `_a` because it is not deterministic
    }
}
