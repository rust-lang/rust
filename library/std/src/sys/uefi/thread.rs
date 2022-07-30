use super::unsupported;
use crate::ffi::CStr;
use crate::io;
use crate::num::NonZeroUsize;
use crate::os::uefi;
use crate::time::Duration;

pub struct Thread(());

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, _p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        unsupported()
    }

    pub fn yield_now() {
        // do nothing
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    pub fn sleep(dur: Duration) {
        if let Some(boot_services) = uefi::env::get_boot_services() {
            let _ = unsafe { ((*boot_services.as_ptr()).stall)(dur.as_micros() as usize) };
        } else {
            panic!("Boot services needed for sleep")
        }
    }

    pub fn join(self) {
        self.0
    }
}

pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    // UEFI is single threaded
    Ok(NonZeroUsize::new(1).unwrap())
}

// FIXME: Should be possible to implement. see https://edk2-docs.gitbook.io/a-tour-beyond-bios-mitigate-buffer-overflow-in-ue/additional_overflow_detection/stack_overflow_detection
pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> {
        None
    }
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}
