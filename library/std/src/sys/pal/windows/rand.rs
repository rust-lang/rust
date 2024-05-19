use crate::sys::c;
use core::mem;
use core::ptr;

#[cfg(not(target_vendor = "win7"))]
#[inline]
pub fn hashmap_random_keys() -> (u64, u64) {
    let mut v = (0, 0);
    let ret = unsafe { c::ProcessPrng(ptr::addr_of_mut!(v).cast::<u8>(), mem::size_of_val(&v)) };
    // ProcessPrng is documented as always returning `TRUE`.
    // https://learn.microsoft.com/en-us/windows/win32/seccng/processprng#return-value
    debug_assert_eq!(ret, c::TRUE);
    v
}

#[cfg(target_vendor = "win7")]
pub fn hashmap_random_keys() -> (u64, u64) {
    use crate::ffi::c_void;
    use crate::io;

    let mut v = (0, 0);
    let ret = unsafe {
        c::RtlGenRandom(ptr::addr_of_mut!(v).cast::<c_void>(), mem::size_of_val(&v) as c::ULONG)
    };

    if ret != 0 { v } else { panic!("RNG broken: {}", io::Error::last_os_error()) }
}
