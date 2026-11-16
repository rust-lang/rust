//@only-target: linux
//@compile-flags: -Zmiri-disable-isolation
//@run-native: ignore-output

use std::fs::File;
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::time::{Duration, SystemTime};

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;
use libc_utils::{errno_check, errno_result};

// glibc's time64 ABI keeps nanoseconds as a 32-bit integer, with endian-dependent padding.
#[repr(C)]
struct Timespec64 {
    tv_sec: i64,
    #[cfg(target_endian = "big")]
    padding: MaybeUninit<i32>,
    tv_nsec: i32,
    #[cfg(target_endian = "little")]
    padding: MaybeUninit<i32>,
}

impl Timespec64 {
    fn new(tv_sec: i64, tv_nsec: i32) -> Self {
        // Leave padding uninitialized so Miri detects if the shim reads it as part of tv_nsec.
        Self { tv_sec, tv_nsec, padding: MaybeUninit::uninit() }
    }
}

/// Test futimens64.
fn main() {
    let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"__futimens64".as_ptr()) };
    if !cfg!(all(
        target_env = "gnu",
        any(target_arch = "x86", target_arch = "arm", target_arch = "mips")
    )) {
        assert!(symbol.is_null());
        return;
    }
    // Native glibc before 2.34 does not provide this symbol.
    if !cfg!(miri) && symbol.is_null() {
        eprintln!("skipping test as __futimens64 symbol is missing");
        return;
    }
    assert!(!symbol.is_null());
    let futimens64: unsafe extern "C" fn(libc::c_int, *const Timespec64) -> libc::c_int =
        unsafe { std::mem::transmute(symbol) };

    let path = utils::prepare_with_content("miri_test_futimens64.txt", b"hello");
    let file = File::options().write(true).open(&path).unwrap();
    let fd = file.as_raw_fd();
    // Legacy fstat has 32-bit timestamps on these targets; std metadata uses statx.
    let get_times = || {
        let metadata = file.metadata().unwrap();
        (metadata.accessed().unwrap(), metadata.modified().unwrap())
    };
    // Also exercise seconds with nonzero upper 32 bits, and fractional seconds in both slots.
    let times =
        [Timespec64::new(5_000_000_000, 100_000_000), Timespec64::new(3_000_000_001, 200_000_000)];
    errno_check(unsafe { futimens64(fd, times.as_ptr()) });
    let [accessed, modified] = times.map(|time| {
        SystemTime::UNIX_EPOCH + Duration::new(time.tv_sec.try_into().unwrap(), time.tv_nsec as u32)
    });
    assert_eq!(get_times(), (accessed, modified));

    // Invalid nanoseconds in either slot must leave both timestamps unchanged.
    for (access_nsec, modified_nsec) in [(-1, 0), (0, -1), (1_000_000_000, 0), (0, 1_000_000_000)] {
        let times = [Timespec64::new(1, access_nsec), Timespec64::new(1, modified_nsec)];
        let err = errno_result(unsafe { futimens64(fd, times.as_ptr()) }).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
        assert_eq!(get_times(), (accessed, modified));
    }

    // Sentinel values ignore tv_sec, including negative seconds.
    let times =
        [Timespec64::new(-1, libc::UTIME_NOW as i32), Timespec64::new(-1, libc::UTIME_OMIT as i32)];
    let before = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
    errno_check(unsafe { futimens64(fd, times.as_ptr()) });
    let (accessed, actual_modified) = get_times();
    assert!(accessed.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() >= before);
    assert_eq!(actual_modified, modified);

    // A null times pointer sets both timestamps to the current time.
    errno_check(unsafe { futimens64(fd, std::ptr::null()) });
    let (accessed, modified) = get_times();
    for time in [accessed, modified] {
        assert!(time.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() >= before);
    }

    // An invalid file descriptor reports EBADF.
    let err = errno_result(unsafe { futimens64(-1, times.as_ptr()) }).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EBADF));
    std::fs::remove_file(path).unwrap();
}
