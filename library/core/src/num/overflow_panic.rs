//! Functions for panicking on overflow.
//!
//! In particular, these are used by the `strict_` methods on integers.

#[cold]
#[track_caller]
pub const fn add() -> ! {
    panic!("attempt to add with overflow")
}

#[cold]
#[track_caller]
pub const fn sub() -> ! {
    panic!("attempt to subtract with overflow")
}

#[cold]
#[track_caller]
pub const fn mul() -> ! {
    panic!("attempt to multiply with overflow")
}

#[cold]
#[track_caller]
pub const fn div() -> ! {
    panic!("attempt to divide with overflow")
}

#[cold]
#[track_caller]
pub const fn rem() -> ! {
    panic!("attempt to calculate the remainder with overflow")
}

#[cold]
#[track_caller]
pub const fn neg() -> ! {
    panic!("attempt to negate with overflow")
}

#[cold]
#[track_caller]
pub const fn shr() -> ! {
    panic!("attempt to shift right with overflow")
}

#[cold]
#[track_caller]
pub const fn shl() -> ! {
    panic!("attempt to shift left with overflow")
}
