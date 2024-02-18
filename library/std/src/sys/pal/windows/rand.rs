use crate::mem;
use crate::ptr;
use crate::sys::c;

pub fn hashmap_random_keys() -> (u64, u64) {
    let mut v = (0, 0);
    let ret = unsafe {
        c::BCryptGenRandom(
            ptr::null_mut(),
            &mut v as *mut _ as *mut u8,
            mem::size_of_val(&v) as c::ULONG,
            c::BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    if c::nt_success(ret) { v } else { fallback_rng() }
}

/// Generate random numbers using the fallback RNG function (RtlGenRandom)
///
/// This is necessary because of a failure to load the SysWOW64 variant of the
/// bcryptprimitives.dll library from code that lives in bcrypt.dll
/// See <https://bugzilla.mozilla.org/show_bug.cgi?id=1788004#c9>
#[cfg(not(target_vendor = "uwp"))]
#[inline(never)]
fn fallback_rng() -> (u64, u64) {
    use crate::ffi::c_void;
    use crate::io;

    let mut v = (0, 0);
    let ret = unsafe {
        c::RtlGenRandom(&mut v as *mut _ as *mut c_void, mem::size_of_val(&v) as c::ULONG)
    };

    if ret != 0 { v } else { panic!("fallback RNG broken: {}", io::Error::last_os_error()) }
}

/// We can't use RtlGenRandom with UWP, so there is no fallback
#[cfg(target_vendor = "uwp")]
#[inline(never)]
fn fallback_rng() -> (u64, u64) {
    panic!("fallback RNG broken: RtlGenRandom() not supported on UWP");
}
