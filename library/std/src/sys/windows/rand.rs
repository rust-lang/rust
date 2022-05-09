use crate::io;
use crate::mem;
use crate::sync;
use crate::sys::c;

// The kinds of HashMap RNG that may be available
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
            fallback_rng().unwrap().expect("couldn't generate random bytes with fallback RNG")
        }
    }
}

// Returns the HashMap RNG that should be used
//
// Panics if they are both broken
fn get_hashmap_rng() -> HashMapRng {
    // Assume that if the preferred RNG is broken the first time we use it, it likely means
    // that: the DLL has failed to load, there is no point to calling it over-and-over again,
    // and we should cache the result
    static INIT: sync::Once = sync::Once::new();
    static mut HASHMAP_RNG: HashMapRng = HashMapRng::Preferred;

    unsafe {
        INIT.call_once(|| HASHMAP_RNG = choose_hashmap_rng());
        HASHMAP_RNG
    }
}

// Test whether we should use the preferred or fallback RNG
//
// If the preferred RNG is successful, we choose it. Otherwise, if the fallback RNG is successful,
// we choose that
//
// Panics if both the preferred and the fallback RNG are both non-functional
fn choose_hashmap_rng() -> HashMapRng {
    let preferred_error = match preferred_rng() {
        Ok(_) => return HashMapRng::Preferred,
        Err(e) => e,
    };

    // On UWP, there is no fallback
    let fallback_result = fallback_rng()
        .unwrap_or_else(|| panic!("preferred RNG broken: `{}`, no fallback", preferred_error));

    match fallback_result {
        Ok(_) => return HashMapRng::Fallback,
        Err(fallback_error) => panic!(
            "preferred RNG broken: `{}`, fallback RNG broken: `{}`",
            preferred_error, fallback_error
        ),
    }
}

// Generate random numbers using the preferred RNG function (BCryptGenRandom)
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

// Generate random numbers using the fallback RNG function (RtlGenRandom)
#[cfg(not(target_vendor = "uwp"))]
fn fallback_rng() -> Option<Result<(u64, u64), io::Error>> {
    let mut v = (0, 0);
    let ret =
        unsafe { c::RtlGenRandom(&mut v as *mut _ as *mut u8, mem::size_of_val(&v) as c::ULONG) };

    Some(if ret != 0 { Ok(v) } else { Err(io::Error::last_os_error()) })
}

// We can't use RtlGenRandom with UWP, so there is no fallback
#[cfg(target_vendor = "uwp")]
fn fallback_rng() -> Option<Result<(u64, u64), io::Error>> {
    None
}
