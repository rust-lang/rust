//@only-target: darwin

use std::cell::UnsafeCell;

fn main() {
    let lock = UnsafeCell::new(libc::OS_UNFAIR_LOCK_INIT);

    unsafe {
        libc::os_unfair_lock_lock(lock.get());
        libc::os_unfair_lock_assert_owner(lock.get());
        assert!(!libc::os_unfair_lock_trylock(lock.get()));
        libc::os_unfair_lock_unlock(lock.get());

        libc::os_unfair_lock_assert_not_owner(lock.get());
    }

    // `os_unfair_lock`s can be moved, and even acquired again then.
    let lock = lock;
    assert!(unsafe { libc::os_unfair_lock_trylock(lock.get()) });
    // We can even move it while locked, but then we cannot acquire it any more.
    let lock = lock;
    assert!(!unsafe { libc::os_unfair_lock_trylock(lock.get()) });
}
