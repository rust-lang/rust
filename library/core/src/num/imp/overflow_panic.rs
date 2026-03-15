//! Functions for panicking on overflow.
//!
//! In particular, these are used by the `strict_` methods on integers.

#[track_caller]
#[rustc_panic_entrypoint]
pub(in crate::num) const fn add() -> ! {
    panic!("attempt to add with overflow")
}

#[track_caller]
#[rustc_panic_entrypoint]
pub(in crate::num) const fn sub() -> ! {
    panic!("attempt to subtract with overflow")
}

#[track_caller]
#[rustc_panic_entrypoint]
pub(in crate::num) const fn mul() -> ! {
    panic!("attempt to multiply with overflow")
}

#[track_caller]
#[rustc_panic_entrypoint]
pub(in crate::num) const fn div() -> ! {
    panic!("attempt to divide with overflow")
}

#[track_caller]
#[rustc_panic_entrypoint]
pub(in crate::num) const fn rem() -> ! {
    panic!("attempt to calculate the remainder with overflow")
}

#[track_caller]
#[rustc_panic_entrypoint]
pub(in crate::num) const fn neg() -> ! {
    panic!("attempt to negate with overflow")
}

#[track_caller]
#[rustc_panic_entrypoint]
pub(in crate::num) const fn shr() -> ! {
    panic!("attempt to shift right with overflow")
}

#[track_caller]
#[rustc_panic_entrypoint]
pub(in crate::num) const fn shl() -> ! {
    panic!("attempt to shift left with overflow")
}
