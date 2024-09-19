use super::unsupported;
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::xous::ffi::Error as XousError;
use crate::path::{self, PathBuf};
use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};
use crate::{fmt, io};

pub(crate) mod params;

static PARAMS_ADDRESS: Atomic<*mut u8> = AtomicPtr::new(core::ptr::null_mut());

#[cfg(not(test))]
#[cfg(feature = "panic_unwind")]
mod eh_unwinding {
    pub(crate) struct EhFrameFinder;
    pub(crate) static mut EH_FRAME_ADDRESS: usize = 0;
    pub(crate) static EH_FRAME_SETTINGS: EhFrameFinder = EhFrameFinder;

    unsafe impl unwind::EhFrameFinder for EhFrameFinder {
        fn find(&self, _pc: usize) -> Option<unwind::FrameInfo> {
            if unsafe { EH_FRAME_ADDRESS == 0 } {
                None
            } else {
                Some(unwind::FrameInfo {
                    text_base: None,
                    kind: unwind::FrameInfoKind::EhFrame(unsafe { EH_FRAME_ADDRESS }),
                })
            }
        }
    }
}

#[cfg(not(test))]
mod c_compat {
    use crate::os::xous::ffi::exit;
    unsafe extern "C" {
        fn main() -> u32;
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn abort() {
        exit(1);
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn _start(eh_frame: usize, params_address: usize) {
        #[cfg(feature = "panic_unwind")]
        {
            unsafe { super::eh_unwinding::EH_FRAME_ADDRESS = eh_frame };
            unwind::set_custom_eh_frame_finder(&super::eh_unwinding::EH_FRAME_SETTINGS).ok();
        }

        if params_address != 0 {
            let params_address = crate::ptr::with_exposed_provenance_mut::<u8>(params_address);
            if unsafe {
                super::params::ApplicationParameters::new_from_ptr(params_address).is_some()
            } {
                super::PARAMS_ADDRESS.store(params_address, core::sync::atomic::Ordering::Relaxed);
            }
        }
        exit(unsafe { main() });
    }

    // This function is needed by the panic runtime. The symbol is named in
    // pre-link args for the target specification, so keep that in sync.
    #[unsafe(no_mangle)]
    // NB. used by both libunwind and libpanic_abort
    pub extern "C" fn __rust_abort() -> ! {
        exit(101);
    }
}

pub fn errno() -> i32 {
    0
}

pub fn error_string(errno: i32) -> String {
    Into::<XousError>::into(errno).to_string()
}

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    unsupported()
}

pub struct SplitPaths<'a>(!, PhantomData<&'a ()>);

pub fn split_paths(_unparsed: &OsStr) -> SplitPaths<'_> {
    panic!("unsupported")
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        self.0
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(_paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    Err(JoinPathsError)
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "not supported on this platform yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "not supported on this platform yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

pub(crate) fn get_application_parameters() -> Option<params::ApplicationParameters> {
    let params_address = PARAMS_ADDRESS.load(Ordering::Relaxed);
    unsafe { params::ApplicationParameters::new_from_ptr(params_address) }
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on this platform")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    crate::os::xous::ffi::exit(code as u32);
}

pub fn getpid() -> u32 {
    panic!("no pids on this platform")
}
