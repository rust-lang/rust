use crate::ffi::CStr;
use crate::io;
use crate::sys::{unsupported, Void};
use crate::time::Duration;

pub struct Thread(Void);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, _p: Box<dyn FnOnce()>)
        -> io::Result<Thread>
    {
        unsupported()
    }

    pub fn yield_now() {
        // do nothing
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    #[cfg(not(target_feature = "atomics"))]
    pub fn sleep(_dur: Duration) {
        panic!("can't sleep");
    }

    #[cfg(target_feature = "atomics")]
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
            let amt = cmp::min(i64::max_value() as u128, nanos);
            let mut x = 0;
            let val = unsafe { wasm32::i32_atomic_wait(&mut x, 0, amt as i64) };
            debug_assert_eq!(val, 2);
            nanos -= amt;
        }
    }

    pub fn join(self) {
        match self.0 {}
    }
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
}

cfg_if::cfg_if! {
    if #[cfg(all(target_feature = "atomics", feature = "wasm-bindgen-threads"))] {
        #[link(wasm_import_module = "__wbindgen_thread_xform__")]
        extern {
            fn __wbindgen_current_id() -> u32;
            fn __wbindgen_tcb_get() -> u32;
            fn __wbindgen_tcb_set(ptr: u32);
        }
        pub fn my_id() -> u32 {
            unsafe { __wbindgen_current_id() }
        }

        // These are currently only ever used in `thread_local_atomics.rs`, if
        // you'd like to use them be sure to update that and make sure everyone
        // agrees what's what.
        pub fn tcb_get() -> *mut u8 {
            use crate::mem;
            assert_eq!(mem::size_of::<*mut u8>(), mem::size_of::<u32>());
            unsafe { __wbindgen_tcb_get() as *mut u8 }
        }

        pub fn tcb_set(ptr: *mut u8) {
            unsafe { __wbindgen_tcb_set(ptr as u32); }
        }

        // FIXME: still need something for hooking exiting a thread to free
        // data...

    } else if #[cfg(target_feature = "atomics")] {
        pub fn my_id() -> u32 {
            panic!("thread ids not implemented on wasm with atomics yet")
        }

        pub fn tcb_get() -> *mut u8 {
            panic!("thread local data not implemented on wasm with atomics yet")
        }

        pub fn tcb_set(_ptr: *mut u8) {
            panic!("thread local data not implemented on wasm with atomics yet")
        }
    } else {
        // stubbed out because no functions actually access these intrinsics
        // unless atomics are enabled
    }
}
