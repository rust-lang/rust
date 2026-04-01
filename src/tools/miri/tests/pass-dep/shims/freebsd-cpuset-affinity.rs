//@only-target: freebsd
//@compile-flags: -Zmiri-num-cpus=256

use std::mem;

fn getaffinity() {
    let mut set: libc::cpuset_t = unsafe { mem::zeroed() };
    unsafe {
        if libc::cpuset_getaffinity(
            libc::CPU_LEVEL_WHICH,
            libc::CPU_WHICH_PID,
            -1,
            size_of::<libc::cpuset_t>(),
            &mut set,
        ) == 0
        {
            assert!(libc::CPU_COUNT(&set) == 256);
        }
    }
}

fn get_small_cpu_mask() {
    let mut set: libc::cpuset_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

    // 256 CPUs so we need 32 bytes to represent this mask.
    // According to Freebsd only when `cpusetsize` is smaller than this value, does it return with ERANGE

    let err = unsafe {
        libc::cpuset_getaffinity(libc::CPU_LEVEL_WHICH, libc::CPU_WHICH_PID, -1, 32, &mut set)
    };
    assert_eq!(err, 0, "Success Expected");

    // 31 is not enough, so it should fail.
    let err = unsafe {
        libc::cpuset_getaffinity(libc::CPU_LEVEL_WHICH, libc::CPU_WHICH_PID, -1, 31, &mut set)
    };
    assert_eq!(err, -1, "Expected Failure");
    assert_eq!(std::io::Error::last_os_error().raw_os_error().unwrap(), libc::ERANGE);

    // Zero should fail as well.
    let err = unsafe {
        libc::cpuset_getaffinity(libc::CPU_LEVEL_WHICH, libc::CPU_WHICH_PID, -1, 0, &mut set)
    };
    assert_eq!(err, -1, "Expected Failure");
    assert_eq!(std::io::Error::last_os_error().raw_os_error().unwrap(), libc::ERANGE);
}

fn main() {
    getaffinity();
    get_small_cpu_mask();
}
