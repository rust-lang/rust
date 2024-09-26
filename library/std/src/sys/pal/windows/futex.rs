use core::ffi::c_void;
use core::sync::atomic::{
    AtomicBool, AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicIsize, AtomicPtr, AtomicU8,
    AtomicU16, AtomicU32, AtomicU64, AtomicUsize,
};
use core::time::Duration;
use core::{mem, ptr};

use super::api::{self, WinError};
use crate::sys::{c, dur2timeout};

/// An atomic for use as a futex that is at least 8-bits but may be larger.
pub type SmallAtomic = AtomicU8;
/// Must be the underlying type of SmallAtomic
pub type SmallPrimitive = u8;

pub unsafe trait Futex {}
pub unsafe trait Waitable {
    type Atomic;
}
macro_rules! unsafe_waitable_int {
    ($(($int:ty, $atomic:ty)),*$(,)?) => {
        $(
            unsafe impl Waitable for $int {
                type Atomic = $atomic;
            }
            unsafe impl Futex for $atomic {}
        )*
    };
}
unsafe_waitable_int! {
    (bool, AtomicBool),
    (i8, AtomicI8),
    (i16, AtomicI16),
    (i32, AtomicI32),
    (i64, AtomicI64),
    (isize, AtomicIsize),
    (u8, AtomicU8),
    (u16, AtomicU16),
    (u32, AtomicU32),
    (u64, AtomicU64),
    (usize, AtomicUsize),
}
unsafe impl<T> Waitable for *const T {
    type Atomic = AtomicPtr<T>;
}
unsafe impl<T> Waitable for *mut T {
    type Atomic = AtomicPtr<T>;
}
unsafe impl<T> Futex for AtomicPtr<T> {}

pub fn wait_on_address<W: Waitable>(
    address: &W::Atomic,
    compare: W,
    timeout: Option<Duration>,
) -> bool {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        let size = mem::size_of::<W>();
        let compare_addr = (&raw const compare).cast::<c_void>();
        let timeout = timeout.map(dur2timeout).unwrap_or(c::INFINITE);
        c::WaitOnAddress(addr, compare_addr, size, timeout) == c::TRUE
    }
}

pub fn wake_by_address_single<T: Futex>(address: &T) {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        c::WakeByAddressSingle(addr);
    }
}

pub fn wake_by_address_all<T: Futex>(address: &T) {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        c::WakeByAddressAll(addr);
    }
}

pub fn futex_wait<W: Waitable>(futex: &W::Atomic, expected: W, timeout: Option<Duration>) -> bool {
    // return false only on timeout
    wait_on_address(futex, expected, timeout) || api::get_last_error() != WinError::TIMEOUT
}

pub fn futex_wake<T: Futex>(futex: &T) -> bool {
    wake_by_address_single(futex);
    false
}

pub fn futex_wake_all<T: Futex>(futex: &T) {
    wake_by_address_all(futex)
}
