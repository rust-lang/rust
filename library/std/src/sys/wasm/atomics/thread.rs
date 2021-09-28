use crate::ffi::CStr;
use crate::io;
use crate::num::NonZeroUsize;
use crate::sys::unsupported;
use crate::time::Duration;

pub struct Thread(!);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, _p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        unsupported()
    }

    pub fn yield_now() {}

    pub fn set_name(_name: &CStr) {}

    pub fn sleep(dur: Duration) {
        use crate::arch::wasm32;
        use crate::cmp;

        // Use an atomic wait to block the current thread artificially with a
        // timeout listed. Note that we should never be notified (return value
        // of 0) or our comparison should never fail (return value of 1) so we
        // should always only resume execution through a timeout (return value
        // 2).
        let mut nanos = dur.as_nanos();
        while nanos > 0 {
            let amt = cmp::min(i64::MAX as u128, nanos);
            let mut x = 0;
            let val = unsafe { wasm32::memory_atomic_wait32(&mut x, 0, amt as i64) };
            debug_assert_eq!(val, 2);
            nanos -= amt;
        }
    }

    pub fn join(self) {}
}

pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    unsupported()
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> {
        None
    }
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}

// We currently just use our own thread-local to store our
// current thread's ID, and then we lazily initialize it to something allocated
// from a global counter.
pub fn my_id() -> u32 {
    use crate::sync::atomic::{AtomicU32, Ordering::SeqCst};

    static NEXT_ID: AtomicU32 = AtomicU32::new(0);

    #[thread_local]
    static mut MY_ID: u32 = 0;

    unsafe {
        // If our thread ID isn't set yet then we need to allocate one. Do so
        // with with a simple "atomically add to a global counter" strategy.
        // This strategy doesn't handled what happens when the counter
        // overflows, however, so just abort everything once the counter
        // overflows and eventually we could have some sort of recycling scheme
        // (or maybe this is all totally irrelevant by that point!). In any case
        // though we're using a CAS loop instead of a `fetch_add` to ensure that
        // the global counter never overflows.
        if MY_ID == 0 {
            let mut cur = NEXT_ID.load(SeqCst);
            MY_ID = loop {
                let next = cur.checked_add(1).unwrap_or_else(|| crate::process::abort());
                match NEXT_ID.compare_exchange(cur, next, SeqCst, SeqCst) {
                    Ok(_) => break next,
                    Err(i) => cur = i,
                }
            };
        }
        MY_ID
    }
}
