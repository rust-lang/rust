use crate::io;
use crate::lazy;
use crate::mem;
use crate::sys::c;

/// The kinds of HashMap RNG that may be available
#[derive(Clone, Copy, Debug, PartialEq)]
enum HashMapRng {
    Preferred,
    Fallback,
}

pub fn hashmap_random_keys() -> (u64, u64) {
    match get_hashmap_rng() {
        HashMapRng::Preferred => {
            preferred_rng().expect("couldn't generate random bytes with preferred RNG")
        }
        HashMapRng::Fallback => {
            fallback_rng().expect("couldn't generate random bytes with fallback RNG")
        }
    }
}

/// Returns the HashMap RNG that should be used
///
/// Panics if they are both broken
fn get_hashmap_rng() -> HashMapRng {
    // Assume that if the preferred RNG is broken the first time we use it, it likely means
    // that: the DLL has failed to load, there is no point to calling it over-and-over again,
    // and we should cache the result
    static VALUE: lazy::SyncOnceCell<HashMapRng> = lazy::SyncOnceCell::new();
    *VALUE.get_or_init(choose_hashmap_rng)
}

/// Test whether we should use the preferred or fallback RNG
///
/// If the preferred RNG is successful, we choose it. Otherwise, if the fallback RNG is successful,
/// we choose that
///
/// Panics if both the preferred and the fallback RNG are both non-functional
fn choose_hashmap_rng() -> HashMapRng {
    let preferred_error = match preferred_rng() {
        Ok(_) => return HashMapRng::Preferred,
        Err(e) => e,
    };

    match fallback_rng() {
        Ok(_) => return HashMapRng::Fallback,
        Err(fallback_error) => panic!(
            "preferred RNG broken: `{}`, fallback RNG broken: `{}`",
            preferred_error, fallback_error
        ),
    }
}

/// Generate random numbers using the preferred RNG function (BCryptGenRandom)
fn preferred_rng() -> Result<(u64, u64), io::Error> {
    use crate::ptr;

    let mut v = (0, 0);
    let ret = unsafe {
        c::BCryptGenRandom(
            ptr::null_mut(),
            &mut v as *mut _ as *mut u8,
            mem::size_of_val(&v) as c::ULONG,
            c::BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };

    if ret == 0 { Ok(v) } else { Err(io::Error::last_os_error()) }
}

/// Generate random numbers using the fallback RNG function (RtlGenRandom)
#[cfg(not(target_vendor = "uwp"))]
fn fallback_rng() -> Result<(u64, u64), io::Error> {
    let mut v = (0, 0);
    let ret =
        unsafe { c::RtlGenRandom(&mut v as *mut _ as *mut u8, mem::size_of_val(&v) as c::ULONG) };

    if ret != 0 { Ok(v) } else { Err(io::Error::last_os_error()) }
}

/// We can't use RtlGenRandom with UWP, so there is no fallback
#[cfg(target_vendor = "uwp")]
fn fallback_rng() -> Result<(u64, u64), io::Error> {
    Err(io::const_io_error!(io::ErrorKind::Unsupported, "unsupported on UWP"))
}
