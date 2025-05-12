use super::unsupported;
use crate::ffi::CStr;
use crate::io;
use crate::num::NonZero;
use crate::ptr::NonNull;
use crate::time::Duration;

pub struct Thread(!);

pub const DEFAULT_MIN_STACK_SIZE: usize = 64 * 1024;

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
        let boot_services: NonNull<r_efi::efi::BootServices> =
            crate::os::uefi::env::boot_services().expect("can't sleep").cast();
        let mut dur_ms = dur.as_micros();
        // ceil up to the nearest microsecond
        if dur.subsec_nanos() % 1000 > 0 {
            dur_ms += 1;
        }

        while dur_ms > 0 {
            let ms = crate::cmp::min(dur_ms, usize::MAX as u128);
            let _ = unsafe { ((*boot_services.as_ptr()).stall)(ms as usize) };
            dur_ms -= ms;
        }
    }

    pub fn join(self) {
        self.0
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    // UEFI is single threaded
    Ok(NonZero::new(1).unwrap())
}
