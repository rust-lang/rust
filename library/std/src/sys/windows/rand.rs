//! # Random key generation
//!
//! This module wraps the RNG provided by the OS. There are a few different
//! ways to interface with the OS RNG so it's worth exploring each of the options.
//! Note that at the time of writing these all go through the (undocumented)
//! `bcryptPrimitives.dll` but they use different route to get there.
//!
//! Originally we were using [`RtlGenRandom`], however that function is
//! deprecated and warns it "may be altered or unavailable in subsequent versions".
//!
//! So we switched to [`BCryptGenRandom`] with the `BCRYPT_USE_SYSTEM_PREFERRED_RNG`
//! flag to query and find the system configured RNG. However, this change caused a small
//! but significant number of users to experience panics caused by a failure of
//! this function. See [#94098].
//!
//! The current version changes this to use the `BCRYPT_RNG_ALG_HANDLE`
//! [Pseudo-handle], which gets the default RNG algorithm without querying the
//! system preference thus hopefully avoiding the previous issue.
//! This is only supported on Windows 10+ so a fallback is used for older versions.
//!
//! [#94098]: https://github.com/rust-lang/rust/issues/94098
//! [`RtlGenRandom`]: https://docs.microsoft.com/en-us/windows/win32/api/ntsecapi/nf-ntsecapi-rtlgenrandom
//! [`BCryptGenRandom`]: https://docs.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom
//! [Pseudo-handle]: https://docs.microsoft.com/en-us/windows/win32/seccng/cng-algorithm-pseudo-handles
use crate::io;
use crate::mem;
use crate::ptr;
use crate::sys::c;

/// Generates high quality secure random keys for use by [`HashMap`].
///
/// This is used to seed the default [`RandomState`].
///
/// [`HashMap`]: crate::collections::HashMap
/// [`RandomState`]: crate::collections::hash_map::RandomState
pub fn hashmap_random_keys() -> (u64, u64) {
    let mut v = (0, 0);
    let ret = unsafe {
        let size = mem::size_of_val(&v).try_into().unwrap();
        c::BCryptGenRandom(
            // BCRYPT_RNG_ALG_HANDLE is only supported in Windows 10+.
            // So for Windows 8.1 and Windows 7 we'll need a fallback when this fails.
            ptr::invalid_mut(c::BCRYPT_RNG_ALG_HANDLE),
            ptr::addr_of_mut!(v).cast(),
            size,
            0,
        )
    };
    if ret != 0 { fallback_rng() } else { v }
}

/// Generate random numbers using the fallback RNG function (RtlGenRandom)
#[cfg(not(target_vendor = "uwp"))]
#[inline(never)]
fn fallback_rng() -> (u64, u64) {
    let mut v = (0, 0);
    let ret =
        unsafe { c::RtlGenRandom(&mut v as *mut _ as *mut u8, mem::size_of_val(&v) as c::ULONG) };

    if ret != 0 { v } else { panic!("fallback RNG broken: {}", io::Error::last_os_error()) }
}

/// We can't use RtlGenRandom with UWP, so there is no fallback
#[cfg(target_vendor = "uwp")]
#[inline(never)]
fn fallback_rng() -> (u64, u64) {
    panic!("fallback RNG broken: RtlGenRandom() not supported on UWP");
}
