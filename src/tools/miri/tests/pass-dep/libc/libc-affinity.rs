//@only-target: linux # these are Linux-specific APIs
//@compile-flags: -Zmiri-disable-isolation -Zmiri-num-cpus=4
#![feature(io_error_more)]
#![feature(pointer_is_aligned_to)]

use std::mem::{size_of, size_of_val};

use libc::{cpu_set_t, sched_getaffinity, sched_setaffinity};

// If pid is zero, then the calling thread is used.
const PID: i32 = 0;

fn null_pointers() {
    let err = unsafe { sched_getaffinity(PID, size_of::<cpu_set_t>(), std::ptr::null_mut()) };
    assert_eq!(err, -1);

    let err = unsafe { sched_setaffinity(PID, size_of::<cpu_set_t>(), std::ptr::null()) };
    assert_eq!(err, -1);
}

fn configure_no_cpus() {
    let cpu_count = std::thread::available_parallelism().unwrap().get();

    let mut cpuset: cpu_set_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

    // configuring no CPUs will fail
    let err = unsafe { sched_setaffinity(PID, size_of::<cpu_set_t>(), &cpuset) };
    assert_eq!(err, -1);
    assert_eq!(std::io::Error::last_os_error().kind(), std::io::ErrorKind::InvalidInput);

    // configuring no (physically available) CPUs will fail
    unsafe { libc::CPU_SET(cpu_count, &mut cpuset) };
    let err = unsafe { sched_setaffinity(PID, size_of::<cpu_set_t>(), &cpuset) };
    assert_eq!(err, -1);
    assert_eq!(std::io::Error::last_os_error().kind(), std::io::ErrorKind::InvalidInput);
}

fn configure_unavailable_cpu() {
    let cpu_count = std::thread::available_parallelism().unwrap().get();

    // Safety: valid value for this type
    let mut cpuset: cpu_set_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

    let err = unsafe { sched_getaffinity(PID, size_of::<cpu_set_t>(), &mut cpuset) };
    assert_eq!(err, 0);

    // by default, only available CPUs are configured
    for i in 0..cpu_count {
        assert!(unsafe { libc::CPU_ISSET(i, &cpuset) });
    }
    assert!(unsafe { !libc::CPU_ISSET(cpu_count, &cpuset) });

    // configure CPU that we don't have
    unsafe { libc::CPU_SET(cpu_count, &mut cpuset) };

    let err = unsafe { sched_setaffinity(PID, size_of::<cpu_set_t>(), &cpuset) };
    assert_eq!(err, 0);

    let err = unsafe { sched_getaffinity(PID, size_of::<cpu_set_t>(), &mut cpuset) };
    assert_eq!(err, 0);

    // the CPU is not set because it is not available
    assert!(!unsafe { libc::CPU_ISSET(cpu_count, &cpuset) });
}

fn large_set() {
    // rust's libc does not currently implement dynamic cpu set allocation
    // and related functions like `CPU_ZERO_S`. So we have to be creative

    // i.e. this has 2048 bits, twice the standard number
    let mut cpuset = [u64::MAX; 32];

    let err = unsafe { sched_setaffinity(PID, size_of_val(&cpuset), cpuset.as_ptr().cast()) };
    assert_eq!(err, 0);

    let err = unsafe { sched_getaffinity(PID, size_of_val(&cpuset), cpuset.as_mut_ptr().cast()) };
    assert_eq!(err, 0);
}

fn get_small_cpu_mask() {
    let mut cpuset: cpu_set_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

    // should be 4 on 32-bit systems and 8 otherwise for systems that implement sched_getaffinity
    let step = size_of::<std::ffi::c_ulong>();

    for i in (0..=2).map(|x| x * step) {
        if i == 0 {
            // 0 always fails
            let err = unsafe { sched_getaffinity(PID, i, &mut cpuset) };
            assert_eq!(err, -1, "fail for {}", i);
            assert_eq!(std::io::Error::last_os_error().kind(), std::io::ErrorKind::InvalidInput);
        } else {
            // other whole multiples of the size of c_ulong works
            let err = unsafe { sched_getaffinity(PID, i, &mut cpuset) };
            assert_eq!(err, 0, "fail for {i}");
        }

        // anything else returns an error
        for j in 1..step {
            let err = unsafe { sched_getaffinity(PID, i + j, &mut cpuset) };
            assert_eq!(err, -1, "success for {}", i + j);
            assert_eq!(std::io::Error::last_os_error().kind(), std::io::ErrorKind::InvalidInput);
        }
    }
}

fn set_small_cpu_mask() {
    let mut cpuset: cpu_set_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

    let err = unsafe { sched_getaffinity(PID, size_of::<cpu_set_t>(), &mut cpuset) };
    assert_eq!(err, 0);

    // setting a mask of size 0 is invalid
    let err = unsafe { sched_setaffinity(PID, 0, &cpuset) };
    assert_eq!(err, -1);
    assert_eq!(std::io::Error::last_os_error().kind(), std::io::ErrorKind::InvalidInput);

    // on LE systems, any other number of bytes (at least up to `size_of<cpu_set_t>()`) will work.
    // on BE systems the CPUs 0..8 are stored in the right-most byte of the first chunk. If that
    // byte is not included, no valid CPUs are configured. We skip those cases.
    let cpu_zero_included_length =
        if cfg!(target_endian = "little") { 1 } else { core::mem::size_of::<std::ffi::c_ulong>() };

    for i in cpu_zero_included_length..24 {
        let err = unsafe { sched_setaffinity(PID, i, &cpuset) };
        assert_eq!(err, 0, "fail for {i}");
    }
}

fn set_custom_cpu_mask() {
    let cpu_count = std::thread::available_parallelism().unwrap().get();

    assert!(cpu_count > 1, "this test cannot do anything interesting with just one thread");

    let mut cpuset: cpu_set_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

    // at the start, thread 1 should be set
    let err = unsafe { sched_getaffinity(PID, size_of::<cpu_set_t>(), &mut cpuset) };
    assert_eq!(err, 0);
    assert!(unsafe { libc::CPU_ISSET(1, &cpuset) });

    // make a valid mask
    unsafe { libc::CPU_ZERO(&mut cpuset) };
    unsafe { libc::CPU_SET(0, &mut cpuset) };

    // giving a smaller mask is fine
    let err = unsafe { sched_setaffinity(PID, 8, &cpuset) };
    assert_eq!(err, 0);

    // and actually disables other threads
    let err = unsafe { sched_getaffinity(PID, size_of::<cpu_set_t>(), &mut cpuset) };
    assert_eq!(err, 0);
    assert!(unsafe { !libc::CPU_ISSET(1, &cpuset) });

    // it is important that we reset the cpu mask now for future tests
    for i in 0..cpu_count {
        unsafe { libc::CPU_SET(i, &mut cpuset) };
    }

    let err = unsafe { sched_setaffinity(PID, size_of::<cpu_set_t>(), &cpuset) };
    assert_eq!(err, 0);
}

fn parent_child() {
    let cpu_count = std::thread::available_parallelism().unwrap().get();

    assert!(cpu_count > 1, "this test cannot do anything interesting with just one thread");

    // configure the parent thread to only run only on CPU 0
    let mut parent_cpuset: cpu_set_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
    unsafe { libc::CPU_SET(0, &mut parent_cpuset) };

    let err = unsafe { sched_setaffinity(PID, size_of::<cpu_set_t>(), &parent_cpuset) };
    assert_eq!(err, 0);

    std::thread::scope(|spawner| {
        spawner.spawn(|| {
            let mut cpuset: cpu_set_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

            let err = unsafe { sched_getaffinity(PID, size_of::<cpu_set_t>(), &mut cpuset) };
            assert_eq!(err, 0);

            // the child inherits its parent's set
            assert!(unsafe { libc::CPU_ISSET(0, &cpuset) });
            assert!(unsafe { !libc::CPU_ISSET(1, &cpuset) });

            // configure cpu 1 for the child
            unsafe { libc::CPU_SET(1, &mut cpuset) };
        });
    });

    let err = unsafe { sched_getaffinity(PID, size_of::<cpu_set_t>(), &mut parent_cpuset) };
    assert_eq!(err, 0);

    // the parent's set should be unaffected
    assert!(unsafe { !libc::CPU_ISSET(1, &parent_cpuset) });

    // it is important that we reset the cpu mask now for future tests
    let mut cpuset = parent_cpuset;
    for i in 0..cpu_count {
        unsafe { libc::CPU_SET(i, &mut cpuset) };
    }

    let err = unsafe { sched_setaffinity(PID, size_of::<cpu_set_t>(), &cpuset) };
    assert_eq!(err, 0);
}

fn main() {
    null_pointers();
    configure_no_cpus();
    configure_unavailable_cpu();
    large_set();
    get_small_cpu_mask();
    set_small_cpu_mask();
    set_custom_cpu_mask();
    parent_child();
}
