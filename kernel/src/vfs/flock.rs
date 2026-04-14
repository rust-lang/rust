//! Kernel-side advisory file locking (flock semantics).
//!
//! This module implements a simple in-kernel advisory lock table.  Locks are
//! tracked per **(inode number, process-id)** pair, mirroring POSIX `flock(2)`
//! semantics where each process holds at most one advisory lock per open file.
//!
//! Supported operations (the `how` argument to [`flock`]):
//!
//! * `LOCK_SH` — acquire a shared (read) lock
//! * `LOCK_EX` — acquire an exclusive (write) lock
//! * `LOCK_NB` — non-blocking flag (can be OR-ed with `LOCK_SH`/`LOCK_EX`)
//! * `LOCK_UN` — release the lock
//!
//! # Limitations
//!
//! * Blocking (waiting) lock acquisition is not yet implemented. When a
//!   blocking request (`LOCK_SH` or `LOCK_EX` without `LOCK_NB`) cannot be
//!   satisfied immediately, `EAGAIN` (`EWOULDBLOCK`) is returned. Full
//!   sleep-wait support is deferred to a future iteration.
//! * Lock upgrade (shared → exclusive while no other holder) is detected and
//!   allowed when the calling process is the sole holder.

use alloc::collections::BTreeMap;
use abi::errors::{Errno, SysResult};
use abi::syscall::flock_flags::{LOCK_EX, LOCK_SH, LOCK_UN};
use spin::Mutex;

/// The type of advisory lock held by one process on one inode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LockType {
    Shared,
    Exclusive,
}

/// Global advisory lock table: (inode, pid) → lock type.
///
/// Each process may hold at most one advisory lock per inode at a time.
/// Acquiring a new lock while already holding one atomically replaces the
/// previous lock (upgrade/downgrade).
static FLOCK_TABLE: Mutex<BTreeMap<(u64, u32), LockType>> = Mutex::new(BTreeMap::new());

/// Apply a `flock(2)`-style lock operation.
///
/// * `ino` — inode number identifying the file
/// * `pid` — caller's process ID
/// * `how` — combination of [`abi::syscall::flock_flags`] values
///
/// # Errors
/// Returns [`Errno::EAGAIN`] (`EWOULDBLOCK`) when the lock cannot be acquired
/// immediately due to a conflicting lock held by another process.
pub fn flock(ino: u64, pid: u32, how: u32) -> SysResult<()> {
    if how & LOCK_UN != 0 {
        release(ino, pid);
        return Ok(());
    }

    let mut table = FLOCK_TABLE.lock();

    if how & LOCK_SH != 0 {
        // Check for a conflicting exclusive lock held by another process.
        if table.iter().any(|((i, p), lt)| *i == ino && *p != pid && *lt == LockType::Exclusive) {
            return Err(Errno::EAGAIN);
        }
        table.insert((ino, pid), LockType::Shared);
        return Ok(());
    }

    if how & LOCK_EX != 0 {
        // Check for any conflicting lock held by another process.
        if table.iter().any(|((i, p), _)| *i == ino && *p != pid) {
            // TODO: sleep the calling thread and retry once contention clears.
            return Err(Errno::EAGAIN);
        }
        table.insert((ino, pid), LockType::Exclusive);
        return Ok(());
    }

    Err(Errno::EINVAL)
}

/// Release any advisory lock held by `pid` on `ino`.
///
/// This is called automatically by [`crate::syscall::handlers::vfs::sys_fs_close`]
/// when a file descriptor is closed, ensuring that stale locks are never left
/// in the table.
pub fn release(ino: u64, pid: u32) {
    FLOCK_TABLE.lock().remove(&(ino, pid));
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi::errors::Errno;
    use abi::syscall::flock_flags::{LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN};

    /// Remove any existing state for the given (ino, pid) pair so tests do
    /// not interfere with each other.
    fn cleanup(ino: u64, pid: u32) {
        FLOCK_TABLE.lock().remove(&(ino, pid));
    }

    #[test]
    fn shared_lock_succeeds_when_no_lock_held() {
        let (ino, pid) = (0xF001, 1);
        cleanup(ino, pid);
        assert_eq!(flock(ino, pid, LOCK_SH), Ok(()));
        release(ino, pid);
    }

    #[test]
    fn multiple_processes_can_hold_shared_locks() {
        let ino = 0xF002;
        cleanup(ino, 10);
        cleanup(ino, 11);
        assert_eq!(flock(ino, 10, LOCK_SH), Ok(()));
        assert_eq!(flock(ino, 11, LOCK_SH), Ok(()));
        release(ino, 10);
        release(ino, 11);
    }

    #[test]
    fn exclusive_lock_succeeds_when_no_lock_held() {
        let (ino, pid) = (0xF003, 2);
        cleanup(ino, pid);
        assert_eq!(flock(ino, pid, LOCK_EX), Ok(()));
        release(ino, pid);
    }

    #[test]
    fn exclusive_lock_fails_when_another_process_holds_shared() {
        let ino = 0xF004;
        cleanup(ino, 20);
        cleanup(ino, 21);
        flock(ino, 20, LOCK_SH).unwrap();
        assert_eq!(flock(ino, 21, LOCK_EX | LOCK_NB), Err(Errno::EAGAIN));
        release(ino, 20);
    }

    #[test]
    fn shared_lock_fails_when_another_process_holds_exclusive() {
        let ino = 0xF005;
        cleanup(ino, 30);
        cleanup(ino, 31);
        flock(ino, 30, LOCK_EX).unwrap();
        assert_eq!(flock(ino, 31, LOCK_SH | LOCK_NB), Err(Errno::EAGAIN));
        release(ino, 30);
    }

    #[test]
    fn unlock_when_no_lock_held_is_noop() {
        let (ino, pid) = (0xF006, 3);
        cleanup(ino, pid);
        assert_eq!(flock(ino, pid, LOCK_UN), Ok(()));
    }

    #[test]
    fn exclusive_lock_fails_when_another_process_holds_exclusive() {
        let ino = 0xF007;
        cleanup(ino, 40);
        cleanup(ino, 41);
        flock(ino, 40, LOCK_EX).unwrap();
        assert_eq!(flock(ino, 41, LOCK_EX | LOCK_NB), Err(Errno::EAGAIN));
        release(ino, 40);
    }

    #[test]
    fn release_removes_entry_from_table() {
        let (ino, pid) = (0xF008, 4);
        cleanup(ino, pid);
        flock(ino, pid, LOCK_EX).unwrap();
        release(ino, pid);
        assert!(!FLOCK_TABLE.lock().contains_key(&(ino, pid)));
    }

    #[test]
    fn invalid_how_returns_einval() {
        let (ino, pid) = (0xF009, 5);
        cleanup(ino, pid);
        assert_eq!(flock(ino, pid, 0), Err(Errno::EINVAL));
    }

    #[test]
    fn same_process_can_upgrade_shared_to_exclusive() {
        let (ino, pid) = (0xF00A, 6);
        cleanup(ino, pid);
        flock(ino, pid, LOCK_SH).unwrap();
        // Upgrade: the caller's own shared lock should not block exclusivity.
        assert_eq!(flock(ino, pid, LOCK_EX), Ok(()));
        release(ino, pid);
    }
}
