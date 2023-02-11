// This is a separate crate so that we can `allow(incomplete_features)` for just `generic_const_exprs`
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(core_intrinsics)]

#[cfg(debug_assertions)]
use std::intrinsics::type_name;
use std::{
    fmt,
    mem::{size_of, transmute_copy, MaybeUninit},
};

#[derive(Copy, Clone)]
pub struct Erased<const N: usize> {
    data: MaybeUninit<[u8; N]>,
    #[cfg(debug_assertions)]
    type_id: &'static str,
}

impl<const N: usize> fmt::Debug for Erased<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Erased<{}>", N)
    }
}

pub type Erase<T> = Erased<{ size_of::<T>() }>;

#[inline(always)]
pub fn erase<T: Copy>(src: T) -> Erased<{ size_of::<T>() }> {
    Erased {
        // SAFETY:: Is it safe to transmute to MaybeUninit
        data: unsafe { transmute_copy(&src) },
        #[cfg(debug_assertions)]
        type_id: type_name::<T>(),
    }
}

/// Restores an erased value.
///
/// This is only safe if `value` is a valid instance of `T`.
/// For example if `T` was erased with `erase` previously.
#[inline(always)]
pub unsafe fn restore<T: Copy>(value: Erased<{ size_of::<T>() }>) -> T {
    #[cfg(debug_assertions)]
    assert_eq!(value.type_id, type_name::<T>());
    unsafe { transmute_copy(&value.data) }
}
