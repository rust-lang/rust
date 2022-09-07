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
    Rng::open().and_then(|rng| rng.gen_random_keys()).unwrap_or_else(fallback_rng)
}

struct Rng(c::BCRYPT_ALG_HANDLE);
impl Rng {
    #[cfg(miri)]
    fn open() -> Result<Self, c::NTSTATUS> {
        const BCRYPT_RNG_ALG_HANDLE: c::BCRYPT_ALG_HANDLE = ptr::invalid_mut(0x81);
        let _ = (
            c::BCryptOpenAlgorithmProvider,
            c::BCryptCloseAlgorithmProvider,
            c::BCRYPT_RNG_ALGORITHM,
            c::STATUS_NOT_SUPPORTED,
        );
        Ok(Self(BCRYPT_RNG_ALG_HANDLE))
    }
    #[cfg(not(miri))]
    // Open a handle to the RNG algorithm.
    fn open() -> Result<Self, c::NTSTATUS> {
        use crate::sync::atomic::AtomicPtr;
        use crate::sync::atomic::Ordering::{Acquire, Release};
        const ERROR_VALUE: c::LPVOID = ptr::invalid_mut(usize::MAX);

        // An atomic is used so we don't need to reopen the handle every time.
        static HANDLE: AtomicPtr<crate::ffi::c_void> = AtomicPtr::new(ptr::null_mut());

        let mut handle = HANDLE.load(Acquire);
        // We use a sentinel value to designate an error occurred last time.
        if handle == ERROR_VALUE {
            Err(c::STATUS_NOT_SUPPORTED)
        } else if handle.is_null() {
            let status = unsafe {
                c::BCryptOpenAlgorithmProvider(
                    &mut handle,
                    c::BCRYPT_RNG_ALGORITHM.as_ptr(),
                    ptr::null(),
                    0,
                )
            };
            if c::nt_success(status) {
                // If another thread opens a handle first then use that handle instead.
                let result = HANDLE.compare_exchange(ptr::null_mut(), handle, Release, Acquire);
                if let Err(previous_handle) = result {
                    // Close our handle and return the previous one.
                    unsafe { c::BCryptCloseAlgorithmProvider(handle, 0) };
                    handle = previous_handle;
                }
                Ok(Self(handle))
            } else {
                HANDLE.store(ERROR_VALUE, Release);
                Err(status)
            }
        } else {
            Ok(Self(handle))
        }
    }

    fn gen_random_keys(self) -> Result<(u64, u64), c::NTSTATUS> {
        let mut v = (0, 0);
        let status = unsafe {
            let size = mem::size_of_val(&v).try_into().unwrap();
            c::BCryptGenRandom(self.0, ptr::addr_of_mut!(v).cast(), size, 0)
        };
        if c::nt_success(status) { Ok(v) } else { Err(status) }
    }
}

/// Generate random numbers using the fallback RNG function (RtlGenRandom)
#[cfg(not(target_vendor = "uwp"))]
#[inline(never)]
fn fallback_rng(rng_status: c::NTSTATUS) -> (u64, u64) {
    let mut v = (0, 0);
    let ret =
        unsafe { c::RtlGenRandom(&mut v as *mut _ as *mut u8, mem::size_of_val(&v) as c::ULONG) };

    if ret != 0 {
        v
    } else {
        panic!(
            "RNG broken: {rng_status:#x}, fallback RNG broken: {}",
            crate::io::Error::last_os_error()
        )
    }
}

/// We can't use RtlGenRandom with UWP, so there is no fallback
#[cfg(target_vendor = "uwp")]
#[inline(never)]
fn fallback_rng(rng_status: c::NTSTATUS) -> (u64, u64) {
    panic!("RNG broken: {rng_status:#x} fallback RNG broken: RtlGenRandom() not supported on UWP");
}
