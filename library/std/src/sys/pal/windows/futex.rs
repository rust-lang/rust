use super::api;
use crate::sys::c;
use crate::sys::dur2timeout;
use core::ffi::c_void;
use core::mem;
use core::ptr;
use core::time::Duration;

#[inline(always)]
pub fn wait_on_address<T, U>(address: &T, compare: U, timeout: Option<Duration>) -> bool {
    assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        let size = mem::size_of::<T>();
        let compare_addr = ptr::addr_of!(compare).cast::<c_void>();
        let timeout = timeout.map(dur2timeout).unwrap_or(c::INFINITE);
        c::WaitOnAddress(addr, compare_addr, size, timeout) == c::TRUE
    }
}

#[inline(always)]
pub fn wake_by_address_single<T>(address: &T) {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        c::WakeByAddressSingle(addr);
    }
}

#[inline(always)]
pub fn wake_by_address_all<T>(address: &T) {
    unsafe {
        let addr = ptr::from_ref(address).cast::<c_void>();
        c::WakeByAddressAll(addr);
    }
}

#[inline(always)]
pub fn futex_wait<T, U>(futex: &T, expected: U, timeout: Option<Duration>) -> bool {
    // return false only on timeout
    wait_on_address(futex, expected, timeout) || api::get_last_error().code != c::ERROR_TIMEOUT
}

#[inline(always)]
pub fn futex_wake<T>(futex: &T) -> bool {
    wake_by_address_single(futex);
    false
}

#[inline(always)]
pub fn futex_wake_all<T>(futex: &T) {
    wake_by_address_all(futex)
}
