use std::hint;
use std::sync::{Mutex, TryLockError, atomic};

fn main() {
    test_mutex_stdlib();
    test_rwlock_stdlib();
    test_spin_loop_hint();
    test_thread_yield_now();
}

fn test_mutex_stdlib() {
    let m = Mutex::new(0);
    {
        let _guard = m.lock();
        assert!(m.try_lock().unwrap_err().would_block());
    }
    drop(m.try_lock().unwrap());
    drop(m);
}

fn test_rwlock_stdlib() {
    use std::sync::RwLock;
    let rw = RwLock::new(0);
    {
        let _read_guard = rw.read().unwrap();
        drop(rw.read().unwrap());
        drop(rw.try_read().unwrap());
        assert!(rw.try_write().unwrap_err().would_block());
    }

    {
        let _write_guard = rw.write().unwrap();
        assert!(rw.try_read().unwrap_err().would_block());
        assert!(rw.try_write().unwrap_err().would_block());
    }
}

trait TryLockErrorExt<T> {
    fn would_block(&self) -> bool;
}

impl<T> TryLockErrorExt<T> for TryLockError<T> {
    fn would_block(&self) -> bool {
        match self {
            TryLockError::WouldBlock => true,
            TryLockError::Poisoned(_) => false,
        }
    }
}

fn test_spin_loop_hint() {
    #[allow(deprecated)]
    atomic::spin_loop_hint();
    hint::spin_loop();
}

fn test_thread_yield_now() {
    std::thread::yield_now();
}
