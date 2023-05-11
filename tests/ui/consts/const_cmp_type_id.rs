// known-bug: #110395
#![feature(const_type_id)]
#![feature(const_trait_impl)]

use std::any::TypeId;

const fn main() {
    assert!(TypeId::of::<u8>() == TypeId::of::<u8>());
    assert!(TypeId::of::<()>() != TypeId::of::<u8>());
    const _A: bool = TypeId::of::<u8>() < TypeId::of::<u16>();
    // can't assert `_A` because it is not deterministic
}
