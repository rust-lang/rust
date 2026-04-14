//! Futex (fast userspace mutex) syscall handlers.
//!
//! Provides kernel-backed wait/wake on a userspace `u32` address.
//! This is the backbone for Rust std's sync primitives (Mutex, Condvar,
//! RwLock, thread parking).

use abi::errors::{Errno, SysResult};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;

use crate::syscall::validate::validate_user_range;

/// A futex wait queue entry: the thread ID that is blocked.
type TaskId = u64;

/// Key for the wait queue: (process address-space id, userspace address).
type FutexKey = (u32, usize);

/// Global futex wait-queue table.
static FUTEX_TABLE: Mutex<BTreeMap<FutexKey, Vec<TaskId>>> = Mutex::new(BTreeMap::new());

fn futex_scope_id() -> u32 {
    if let Some(pinfo) = crate::sched::process_info_current() {
        pinfo.lock().pid
    } else {
        unsafe { crate::sched::current_tid_current() as u32 }
    }
}

fn futex_key_for_scope(scope_id: u32, uaddr: usize) -> FutexKey {
    (scope_id, uaddr)
}

fn futex_key(uaddr: usize) -> FutexKey {
    futex_key_for_scope(futex_scope_id(), uaddr)
}

/// `sys_futex_wait(uaddr, expected, timeout_ns)`
///
/// If `*uaddr == expected`, block the current thread until woken or
/// until `timeout_ns` nanoseconds elapse (0 = wait forever).
///
/// Returns:
/// - `Ok(0)` on wake or spurious wakeup
/// - `Err(EAGAIN)` if `*uaddr != expected` (value changed)
/// - `Err(ETIMEDOUT)` on timeout
pub fn sys_futex_wait(uaddr: usize, expected: u32, timeout_ns: u64) -> SysResult<usize> {
    if crate::sched::take_pending_interrupt_current() {
        return Err(Errno::EINTR);
    }

    // Validate the userspace address is readable.
    validate_user_range(uaddr, 4, false)?;

    // Read the current value at the userspace address.
    let current_val = unsafe { core::ptr::read_volatile(uaddr as *const u32) };

    if current_val != expected {
        return Err(Errno::EAGAIN);
    }

    let tid = unsafe { crate::sched::current_tid_current() };
    let key = futex_key(uaddr);

    // Add ourselves to the wait queue.
    {
        let mut table = FUTEX_TABLE.lock();
        table.entry(key).or_insert_with(Vec::new).push(tid);
    }

    if timeout_ns == 0 {
        // Indefinite wait — block until woken.
        unsafe {
            crate::sched::block_current_erased();
        }
        let mut table = FUTEX_TABLE.lock();
        if let Some(waiters) = table.get_mut(&key) {
            if let Some(pos) = waiters.iter().position(|&w| w == tid) {
                waiters.remove(pos);
                if waiters.is_empty() {
                    table.remove(&key);
                }
            }
        }
        if crate::sched::take_pending_interrupt_current() {
            return Err(Errno::EINTR);
        }
    } else {
        // Timed wait — use scheduler sleep with the same coarse rounding as SYS_SLEEP.
        let ticks = crate::time::duration_to_sleep_ticks(timeout_ns);
        if ticks == 0 {
            unsafe {
                crate::sched::yield_now_current();
            }
        } else {
            crate::sched::sleep_ticks_current(ticks);
        }

        if crate::sched::take_pending_interrupt_current() {
            let mut table = FUTEX_TABLE.lock();
            if let Some(waiters) = table.get_mut(&key) {
                if let Some(pos) = waiters.iter().position(|&w| w == tid) {
                    waiters.remove(pos);
                    if waiters.is_empty() {
                        table.remove(&key);
                    }
                }
            }
            return Err(Errno::EINTR);
        }

        // After waking, remove ourselves from the wait queue if still there
        // (we may have been woken by the timer, not by futex_wake).
        let mut table = FUTEX_TABLE.lock();
        if let Some(waiters) = table.get_mut(&key) {
            if let Some(pos) = waiters.iter().position(|&w| w == tid) {
                waiters.remove(pos);
                if waiters.is_empty() {
                    table.remove(&key);
                }
                // We timed out (still in queue = nobody woke us).
                return Err(Errno::ETIMEDOUT);
            }
        }
        // If we're not in the queue, we were woken by futex_wake — success.
    }

    Ok(0)
}

/// `sys_futex_wake(uaddr, count)`
///
/// Wake up to `count` threads waiting on `uaddr`.
/// Returns the number of threads actually woken.
pub fn sys_futex_wake(uaddr: usize, count: u32) -> SysResult<usize> {
    let mut woken = 0u32;
    let key = futex_key(uaddr);

    let mut table = FUTEX_TABLE.lock();
    if let Some(waiters) = table.get_mut(&key) {
        while woken < count {
            if let Some(tid) = waiters.pop() {
                unsafe {
                    crate::sched::wake_task_erased(tid);
                }
                woken += 1;
            } else {
                break;
            }
        }
        if waiters.is_empty() {
            table.remove(&key);
        }
    }

    Ok(woken as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn futex_key_includes_scope() {
        assert_eq!(futex_key_for_scope(7, 0x1000), (7, 0x1000));
    }
}
