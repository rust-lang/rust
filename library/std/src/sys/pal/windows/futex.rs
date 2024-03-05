use super::api;
use crate::sys::c;
use crate::sys::dur2timeout;
use core::ffi::c_void;
use core::mem;
use core::ptr;
use core::sync::atomic::{
    AtomicBool, AtomicI16, AtomicI32, AtomicI64, AtomicI8, AtomicIsize, AtomicPtr, AtomicU16,
    AtomicU32, AtomicU64, AtomicU8, AtomicUsize,
};
use core::time::Duration;

pub unsafe trait Waitable {
    type Atomic;
}
macro_rules! unsafe_waitable_int {
    ($(($int:ty, $atomic:ty)),*$(,)?) => {
        $(
            unsafe impl Waitable for $int {
                type Atomic = $atomic;
            }
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

pub fn wait_on_address<W: Waitable>(
    address: &W::Atomic,
    compare: W,
    timeout: Option<Duration>,
) -> bool {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        let size = mem::size_of::<W>();
        let compare_addr = ptr::addr_of!(compare).cast::<c_void>();
        let timeout = timeout.map(dur2timeout).unwrap_or(c::INFINITE);
        c::WaitOnAddress(addr, compare_addr, size, timeout) == c::TRUE
    }
}

pub fn wake_by_address_single<T>(address: &T) {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        c::WakeByAddressSingle(addr);
    }
}

pub fn wake_by_address_all<T>(address: &T) {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        c::WakeByAddressAll(addr);
    }
}

pub fn futex_wait<W: Waitable>(futex: &W::Atomic, expected: W, timeout: Option<Duration>) -> bool {
    // return false only on timeout
    wait_on_address(futex, expected, timeout) || api::get_last_error().code != c::ERROR_TIMEOUT
}

pub fn futex_wake<T>(futex: &T) -> bool {
    wake_by_address_single(futex);
    false
}

pub fn futex_wake_all<T>(futex: &T) {
    wake_by_address_all(futex)
}
