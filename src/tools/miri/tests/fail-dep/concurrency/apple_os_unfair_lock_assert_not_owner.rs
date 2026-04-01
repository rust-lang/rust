//@only-target: darwin

use std::cell::UnsafeCell;

fn main() {
    let lock = UnsafeCell::new(libc::OS_UNFAIR_LOCK_INIT);

    unsafe {
        libc::os_unfair_lock_lock(lock.get());
        libc::os_unfair_lock_assert_not_owner(lock.get());
        //~^ error: abnormal termination: called os_unfair_lock_assert_not_owner on an os_unfair_lock owned by the current thread
    }
}
