use rustc_const_eval::interpret::{InterpResult, interp_ok};

static SUPERVISOR: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub struct Supervisor;

#[derive(Debug)]
pub struct SvInitError;

impl Supervisor {
    #[inline(always)]
    pub fn is_enabled() -> bool {
        false
    }

    pub fn do_ffi<'tcx, T, U>(
        _: T,
        f: impl FnOnce() -> U,
    ) -> InterpResult<'tcx, (U, Option<super::MemEvents>)> {
        // We acquire the lock to ensure that no two FFI calls run concurrently.
        let _g = SUPERVISOR.lock().unwrap();
        interp_ok((f(), None))
    }
}

#[inline(always)]
#[allow(dead_code, clippy::missing_safety_doc)]
pub unsafe fn init_sv() -> Result<!, SvInitError> {
    Err(SvInitError)
}

#[inline(always)]
#[allow(dead_code)]
pub fn register_retcode_sv<T>(_: T) {}
