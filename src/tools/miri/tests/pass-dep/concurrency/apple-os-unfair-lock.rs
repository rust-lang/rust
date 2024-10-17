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

    // `os_unfair_lock`s can be moved and leaked.
    // In the real implementation, even moving it while locked is possible
    // (and "forks" the lock, i.e. old and new location have independent wait queues).
    // We only test the somewhat sane case of moving while unlocked that `std` plans to rely on.
    let lock = lock;
    let locked = unsafe { libc::os_unfair_lock_trylock(lock.get()) };
    assert!(locked);
    let _lock = lock;
}
