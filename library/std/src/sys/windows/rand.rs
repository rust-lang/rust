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
//! The current version falls back to using `BCryptOpenAlgorithmProvider` if
//! `BCRYPT_USE_SYSTEM_PREFERRED_RNG` fails for any reason.
//!
//! [#94098]: https://github.com/rust-lang/rust/issues/94098
//! [`RtlGenRandom`]: https://docs.microsoft.com/en-us/windows/win32/api/ntsecapi/nf-ntsecapi-rtlgenrandom
//! [`BCryptGenRandom`]: https://docs.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom
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
    Rng::SYSTEM.gen_random_keys().unwrap_or_else(fallback_rng)
}

struct Rng {
    algorithm: c::BCRYPT_ALG_HANDLE,
    flags: u32,
}
impl Rng {
    const SYSTEM: Self = unsafe { Self::new(ptr::null_mut(), c::BCRYPT_USE_SYSTEM_PREFERRED_RNG) };

    /// Create the RNG from an existing algorithm handle.
    ///
    /// # Safety
    ///
    /// The handle must either be null or a valid algorithm handle.
    const unsafe fn new(algorithm: c::BCRYPT_ALG_HANDLE, flags: u32) -> Self {
        Self { algorithm, flags }
    }

    /// Open a handle to the RNG algorithm.
    fn open() -> Result<Self, c::NTSTATUS> {
        use crate::sync::atomic::AtomicPtr;
        use crate::sync::atomic::Ordering::{Acquire, Release};

        // An atomic is used so we don't need to reopen the handle every time.
        static HANDLE: AtomicPtr<crate::ffi::c_void> = AtomicPtr::new(ptr::null_mut());

        let mut handle = HANDLE.load(Acquire);
        if handle.is_null() {
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
                Ok(unsafe { Self::new(handle, 0) })
            } else {
                Err(status)
            }
        } else {
            Ok(unsafe { Self::new(handle, 0) })
        }
    }

    fn gen_random_keys(self) -> Result<(u64, u64), c::NTSTATUS> {
        let mut v = (0, 0);
        let status = unsafe {
            let size = mem::size_of_val(&v).try_into().unwrap();
            c::BCryptGenRandom(self.algorithm, ptr::addr_of_mut!(v).cast(), size, self.flags)
        };
        if c::nt_success(status) { Ok(v) } else { Err(status) }
    }
}

/// Generate random numbers using the fallback RNG function
#[inline(never)]
fn fallback_rng(rng_status: c::NTSTATUS) -> (u64, u64) {
    match Rng::open().and_then(|rng| rng.gen_random_keys()) {
        Ok(keys) => keys,
        Err(status) => {
            panic!("RNG broken: {rng_status:#x}, fallback RNG broken: {status:#x}")
        }
    }
}
