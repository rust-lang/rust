use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{
    Atomic, AtomicBool, AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicIsize, AtomicPtr,
    AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize,
};
use core::time::Duration;

use super::api::{self, WinError};
use crate::sys::{c, dur2timeout};

/// An atomic for use as a futex that is at least 32-bits but may be larger
pub type Futex = Atomic<Primitive>;
/// Must be the underlying type of Futex
pub type Primitive = u32;

/// An atomic for use as a futex that is at least 8-bits but may be larger.
pub type SmallFutex = Atomic<SmallPrimitive>;
/// Must be the underlying type of SmallFutex
pub type SmallPrimitive = u8;

pub unsafe trait Futexable {}
pub unsafe trait Waitable {
    type Futex;
}
macro_rules! unsafe_waitable_int {
    ($(($int:ty, $atomic:ty)),*$(,)?) => {
        $(
            unsafe impl Waitable for $int {
                type Futex = $atomic;
            }
            unsafe impl Futexable for $atomic {}
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
    type Futex = Atomic<*mut T>;
}
unsafe impl<T> Waitable for *mut T {
    type Futex = Atomic<*mut T>;
}
unsafe impl<T> Futexable for AtomicPtr<T> {}

pub fn wait_on_address<W: Waitable>(
    address: &W::Futex,
    compare: W,
    timeout: Option<Duration>,
) -> bool {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        let size = size_of::<W>();
        let compare_addr = (&raw const compare).cast::<c_void>();
        let timeout = timeout.map(dur2timeout).unwrap_or(c::INFINITE);
        c::WaitOnAddress(addr, compare_addr, size, timeout) == c::TRUE
    }
}

pub fn wake_by_address_single<T: Futexable>(address: &T) {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        c::WakeByAddressSingle(addr);
    }
}

pub fn wake_by_address_all<T: Futexable>(address: &T) {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        c::WakeByAddressAll(addr);
    }
}

pub fn futex_wait<W: Waitable>(futex: &W::Futex, expected: W, timeout: Option<Duration>) -> bool {
    // return false only on timeout
    wait_on_address(futex, expected, timeout) || api::get_last_error() != WinError::TIMEOUT
}

pub fn futex_wake<T: Futexable>(futex: &T) -> bool {
    wake_by_address_single(futex);
    false
}

pub fn futex_wake_all<T: Futexable>(futex: &T) {
    wake_by_address_all(futex)
}
