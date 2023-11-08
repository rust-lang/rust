use super::mem;
use crate::arch::asm;

extern "C" {
    static TLS_INIT_BASE: u64;
    static TLS_INIT_SIZE: usize;
    static TLS_OFFSET: usize;
}

pub struct Tls {}

impl Tls {
    /// Initialize the thread local storage to a fresh state.
    ///
    /// # Safety
    /// * may only be called once per thread
    /// * must be dropped before thread exit
    /// * must be called before any `#[thread_local]` variable is used
    pub unsafe fn init() -> Tls {
        // The thread pointer points to the end of the TLS section. It is stored
        // both in the `fs` segment base address (initialized by the loader) and
        // at the address it itself points to (initialized during TCS
        // initialization).
        let tp: *mut u8;
        unsafe {
            asm!("mov fs:0, {}", out(reg) tp, options(preserves_flags, readonly));
        }

        // Initialize the TLS data.
        unsafe {
            let init_base = mem::rel_ptr_mut(TLS_INIT_BASE);
            // The first `TLS_INIT_SIZE` bytes of the TLS section hold non-trivial
            // data that needs to be copied from the initialization image.
            tp.sub(TLS_OFFSET).copy_from_nonoverlapping(init_base, TLS_INIT_SIZE);
            // All remaining bytes are initialized to zero.
            tp.sub(TLS_OFFSET).add(TLS_INIT_SIZE).write_bytes(0, TLS_OFFSET - TLS_INIT_SIZE);
        }

        Tls {}
    }
}

impl Drop for Tls {
    fn drop(&mut self) {
        unsafe {
            crate::sys::thread_local_dtor::run_dtors();
        }
    }
}
