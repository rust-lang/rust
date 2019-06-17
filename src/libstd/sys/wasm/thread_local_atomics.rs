use crate::sys::thread;
use crate::sync::atomic::{AtomicUsize, Ordering::SeqCst};

const MAX_KEYS: usize = 128;
static NEXT_KEY: AtomicUsize = AtomicUsize::new(0);

struct ThreadControlBlock {
    keys: [*mut u8; MAX_KEYS],
}

impl ThreadControlBlock {
    fn new() -> ThreadControlBlock {
        ThreadControlBlock {
            keys: [core::ptr::null_mut(); MAX_KEYS],
        }
    }

    fn get() -> *mut ThreadControlBlock {
        let ptr = thread::tcb_get();
        if !ptr.is_null() {
            return ptr as *mut ThreadControlBlock
        }
        let tcb = Box::into_raw(Box::new(ThreadControlBlock::new()));
        thread::tcb_set(tcb as *mut u8);
        tcb
    }
}

pub type Key = usize;

pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    drop(dtor); // FIXME: need to figure out how to hook thread exit to run this
    let key = NEXT_KEY.fetch_add(1, SeqCst);
    if key >= MAX_KEYS {
        NEXT_KEY.store(MAX_KEYS, SeqCst);
        panic!("cannot allocate space for more TLS keys");
    }
    // offset by 1 so we never hand out 0. This is currently required by
    // `sys_common/thread_local.rs` where it can't cope with keys of value 0
    // because it messes up the atomic management.
    return key + 1
}

pub unsafe fn set(key: Key, value: *mut u8) {
    (*ThreadControlBlock::get()).keys[key - 1] = value;
}

pub unsafe fn get(key: Key) -> *mut u8 {
    (*ThreadControlBlock::get()).keys[key - 1]
}

pub unsafe fn destroy(_key: Key) {
    // FIXME: should implement this somehow, this isn't typically called but it
    // can be called if two threads race to initialize a TLS slot and one ends
    // up not being needed.
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
