// build-pass (FIXME(62277): could be check-pass?)

use std::cell::UnsafeCell;
use std::sync::atomic::AtomicU32;
pub struct Condvar {
    condvar: UnsafeCell<AtomicU32>,
}

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct NoWait(u32);

const CONDVAR_HAS_NO_WAITERS: NoWait = NoWait(42);

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar {
            condvar: UnsafeCell::new(AtomicU32::new(CONDVAR_HAS_NO_WAITERS.0)),
        }
    }
}

fn main() {}
