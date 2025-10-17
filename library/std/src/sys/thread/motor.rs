use crate::ffi::CStr;
use crate::io;
use crate::num::NonZeroUsize;
use crate::sys::map_motor_error;
use crate::time::Duration;

pub const DEFAULT_MIN_STACK_SIZE: usize = 1024 * 256;

pub struct Thread {
    sys_thread: moto_rt::thread::ThreadHandle,
}

unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

impl Thread {
    pub unsafe fn new(
        stack: usize,
        _name: Option<&str>,
        p: Box<dyn FnOnce()>,
    ) -> io::Result<Thread> {
        extern "C" fn __moto_rt_thread_fn(thread_arg: u64) {
            unsafe {
                Box::from_raw(
                    core::ptr::with_exposed_provenance::<Box<dyn FnOnce()>>(thread_arg as usize)
                        .cast_mut(),
                )();
            }
        }

        let thread_arg = Box::into_raw(Box::new(p)).expose_provenance() as u64;
        let sys_thread = moto_rt::thread::spawn(__moto_rt_thread_fn, stack, thread_arg)
            .map_err(map_motor_error)?;
        Ok(Self { sys_thread })
    }

    pub fn join(self) {
        assert!(moto_rt::thread::join(self.sys_thread) == moto_rt::E_OK)
    }
}

pub fn set_name(name: &CStr) {
    let bytes = name.to_bytes();
    if let Ok(s) = core::str::from_utf8(bytes) {
        let _ = moto_rt::thread::set_name(s);
    }
}

pub fn current_os_id() -> Option<u64> {
    None
}

pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    Ok(unsafe { NonZeroUsize::new_unchecked(moto_rt::num_cpus()) })
}

pub fn yield_now() {
    moto_rt::thread::yield_now()
}

pub fn sleep(dur: Duration) {
    moto_rt::thread::sleep_until(moto_rt::time::Instant::now() + dur)
}
