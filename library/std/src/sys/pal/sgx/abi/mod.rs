#![cfg_attr(test, allow(unused))] // RT initialization logic is not compiled for test

use core::arch::global_asm;
use core::sync::atomic::{Atomic, AtomicUsize, Ordering};

use crate::io::Write;

// runtime features
pub mod panic;
mod reloc;

// library features
pub mod mem;
pub mod thread;
pub mod tls;
#[macro_use]
pub mod usercalls;

#[cfg(not(test))]
global_asm!(include_str!("entry.S"), options(att_syntax));

#[repr(C)]
struct EntryReturn(u64, u64);

#[cfg(not(test))]
#[unsafe(no_mangle)]
unsafe extern "C" fn tcs_init(secondary: bool) {
    // Be very careful when changing this code: it runs before the binary has been
    // relocated. Any indirect accesses to symbols will likely fail.
    const UNINIT: usize = 0;
    const BUSY: usize = 1;
    const DONE: usize = 2;
    // Three-state spin-lock
    static RELOC_STATE: Atomic<usize> = AtomicUsize::new(UNINIT);

    if secondary && RELOC_STATE.load(Ordering::Relaxed) != DONE {
        rtabort!("Entered secondary TCS before main TCS!")
    }

    // Try to atomically swap UNINIT with BUSY. The returned state can be:
    match RELOC_STATE.compare_exchange(UNINIT, BUSY, Ordering::Acquire, Ordering::Acquire) {
        // This thread just obtained the lock and other threads will observe BUSY
        Ok(_) => {
            reloc::relocate_elf_rela();
            RELOC_STATE.store(DONE, Ordering::Release);
        }
        // We need to wait until the initialization is done.
        Err(BUSY) => {
            while RELOC_STATE.load(Ordering::Acquire) == BUSY {
                core::hint::spin_loop();
            }
        }
        // Initialization is done.
        Err(DONE) => {}
        _ => unreachable!(),
    }
}

// FIXME: this item should only exist if this is linked into an executable
// (main function exists). If this is a library, the crate author should be
// able to specify this
#[cfg(not(test))]
#[unsafe(no_mangle)]
extern "C" fn entry(p1: u64, p2: u64, p3: u64, secondary: bool, p4: u64, p5: u64) -> EntryReturn {
    // FIXME: how to support TLS in library mode?
    let tls = Box::new(tls::Tls::new());
    let tls_guard = unsafe { tls.activate() };

    if secondary {
        let join_notifier = crate::sys::thread::Thread::entry();
        drop(tls_guard);
        drop(join_notifier);

        EntryReturn(0, 0)
    } else {
        unsafe extern "C" {
            fn main(argc: isize, argv: *const *const u8) -> isize;
        }

        // check entry is being called according to ABI
        rtassert!(p3 == 0);
        rtassert!(p4 == 0);
        rtassert!(p5 == 0);

        unsafe {
            // The actual types of these arguments are `p1: *const Arg, p2:
            // usize`. We can't currently customize the argument list of Rust's
            // main function, so we pass these in as the standard pointer-sized
            // values in `argc` and `argv`.
            let ret = main(p2 as _, p1 as _);
            exit_with_code(ret)
        }
    }
}

pub(super) fn exit_with_code(code: isize) -> ! {
    if code != 0 {
        if let Some(mut out) = panic::SgxPanicOutput::new() {
            let _ = write!(out, "Exited with status code {code}");
        }
    }
    usercalls::exit(code != 0);
}

#[cfg(not(test))]
#[unsafe(no_mangle)]
extern "C" fn abort_reentry() -> ! {
    usercalls::exit(false)
}
