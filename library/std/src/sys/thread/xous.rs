use core::arch::asm;

use crate::io;
use crate::num::NonZero;
use crate::os::xous::ffi::{
    MemoryFlags, Syscall, ThreadId, blocking_scalar, create_thread, do_yield, join_thread,
    map_memory, update_memory_flags,
};
use crate::os::xous::services::{TicktimerScalar, ticktimer_server};
use crate::time::Duration;

pub struct Thread {
    tid: ThreadId,
}

pub const DEFAULT_MIN_STACK_SIZE: usize = 131072;
const MIN_STACK_SIZE: usize = 4096;
pub const GUARD_PAGE_SIZE: usize = 4096;

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(
        stack: usize,
        _name: Option<&str>,
        p: Box<dyn FnOnce()>,
    ) -> io::Result<Thread> {
        let p = Box::into_raw(Box::new(p));
        let mut stack_size = crate::cmp::max(stack, MIN_STACK_SIZE);

        if (stack_size & 4095) != 0 {
            stack_size = (stack_size + 4095) & !4095;
        }

        // Allocate the whole thing, then divide it up after the fact. This ensures that
        // even if there's a context switch during this function, the whole stack plus
        // guard pages will remain contiguous.
        let stack_plus_guard_pages: &mut [u8] = unsafe {
            map_memory(
                None,
                None,
                GUARD_PAGE_SIZE + stack_size + GUARD_PAGE_SIZE,
                MemoryFlags::R | MemoryFlags::W | MemoryFlags::X,
            )
        }
        .map_err(|code| io::Error::from_raw_os_error(code as i32))?;

        // No access to this page. Note: Write-only pages are illegal, and will
        // cause an access violation.
        unsafe {
            update_memory_flags(&mut stack_plus_guard_pages[0..GUARD_PAGE_SIZE], MemoryFlags::W)
                .map_err(|code| io::Error::from_raw_os_error(code as i32))?
        };

        // No access to this page. Note: Write-only pages are illegal, and will
        // cause an access violation.
        unsafe {
            update_memory_flags(
                &mut stack_plus_guard_pages[(GUARD_PAGE_SIZE + stack_size)..],
                MemoryFlags::W,
            )
            .map_err(|code| io::Error::from_raw_os_error(code as i32))?
        };

        let guard_page_pre = stack_plus_guard_pages.as_ptr() as usize;
        let tid = create_thread(
            thread_start as *mut usize,
            &mut stack_plus_guard_pages[GUARD_PAGE_SIZE..(stack_size + GUARD_PAGE_SIZE)],
            p as usize,
            guard_page_pre,
            stack_size,
            0,
        )
        .map_err(|code| io::Error::from_raw_os_error(code as i32))?;

        extern "C" fn thread_start(
            main: *mut usize,
            guard_page_pre: usize,
            stack_size: usize,
        ) -> ! {
            unsafe {
                // Run the contents of the new thread.
                Box::from_raw(main as *mut Box<dyn FnOnce()>)();
            }

            // Destroy TLS, which will free the TLS page and call the destructor for
            // any thread local storage (if any).
            unsafe {
                crate::sys::thread_local::key::destroy_tls();
            }

            // Deallocate the stack memory, along with the guard pages. Afterwards,
            // exit the thread by returning to the magic address 0xff80_3000usize,
            // which tells the kernel to deallocate this thread.
            let mapped_memory_base = guard_page_pre;
            let mapped_memory_length = GUARD_PAGE_SIZE + stack_size + GUARD_PAGE_SIZE;
            unsafe {
                asm!(
                    "ecall",
                    "ret",
                                        in("a0") Syscall::UnmapMemory as usize,
                                        in("a1") mapped_memory_base,
                                        in("a2") mapped_memory_length,
                                        in("ra") 0xff80_3000usize,
                                        options(nomem, nostack, noreturn)
                );
            }
        }

        Ok(Thread { tid })
    }

    pub fn join(self) {
        join_thread(self.tid).unwrap();
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    // We're unicore right now.
    Ok(unsafe { NonZero::new_unchecked(1) })
}

pub fn yield_now() {
    do_yield();
}

pub fn sleep(dur: Duration) {
    // Because the sleep server works on units of `usized milliseconds`, split
    // the messages up into these chunks. This means we may run into issues
    // if you try to sleep a thread for more than 49 days on a 32-bit system.
    let mut millis = dur.as_millis();
    while millis > 0 {
        let sleep_duration = if millis > (usize::MAX as _) { usize::MAX } else { millis as usize };
        blocking_scalar(ticktimer_server(), TicktimerScalar::SleepMs(sleep_duration).into())
            .expect("failed to send message to ticktimer server");
        millis -= sleep_duration as u128;
    }
}
