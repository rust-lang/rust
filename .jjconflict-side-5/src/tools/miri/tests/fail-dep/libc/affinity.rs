//@only-target: linux # these are Linux-specific APIs
//@compile-flags: -Zmiri-disable-isolation -Zmiri-num-cpus=4

fn main() {
    use std::mem::size_of;

    use libc::{cpu_set_t, sched_setaffinity};

    // If pid is zero, then the calling thread is used.
    const PID: i32 = 0;

    let cpuset: cpu_set_t = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

    let err = unsafe { sched_setaffinity(PID, size_of::<cpu_set_t>() + 1, &cpuset) }; //~ ERROR: memory access failed
    assert_eq!(err, 0);
}
