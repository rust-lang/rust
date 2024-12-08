use super::unsupported;
use crate::collections::HashMap;
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::xous::ffi::Error as XousError;
use crate::path::{self, PathBuf};
use crate::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use crate::sync::{Mutex, Once};
use crate::{fmt, io, vec};

pub(crate) mod params;

static PARAMS_ADDRESS: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());

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
    extern "C" {
        fn main() -> u32;
    }

    #[no_mangle]
    pub extern "C" fn abort() {
        exit(1);
    }

    #[no_mangle]
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
    #[no_mangle]
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

// ---------- Environment handling ---------- //
static ENV: AtomicUsize = AtomicUsize::new(0);
static ENV_INIT: Once = Once::new();
type EnvStore = Mutex<HashMap<OsString, OsString>>;

fn get_env_store() -> &'static EnvStore {
    ENV_INIT.call_once(|| {
        let env_store = EnvStore::default();
        if let Some(params) = get_application_parameters() {
            for param in params {
                if let Ok(envs) = params::EnvironmentBlock::try_from(&param) {
                    let mut env_store = env_store.lock().unwrap();
                    for env in envs {
                        env_store.insert(env.key.into(), env.value.into());
                    }
                    break;
                }
            }
        }
        ENV.store(Box::into_raw(Box::new(env_store)) as _, Ordering::Relaxed)
    });
    unsafe { &*core::ptr::with_exposed_provenance::<EnvStore>(ENV.load(Ordering::Relaxed)) }
}

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
}

// FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
pub struct EnvStrDebug<'a> {
    slice: &'a [(OsString, OsString)],
}

impl fmt::Debug for EnvStrDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { slice } = self;
        f.debug_list()
            .entries(slice.iter().map(|(a, b)| (a.to_str().unwrap(), b.to_str().unwrap())))
            .finish()
    }
}

impl Env {
    // FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
    pub fn str_debug(&self) -> impl fmt::Debug + '_ {
        let Self { iter } = self;
        EnvStrDebug { slice: iter.as_slice() }
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { iter } = self;
        f.debug_list().entries(iter.as_slice()).finish()
    }
}

impl !Send for Env {}
impl !Sync for Env {}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub fn env() -> Env {
    let clone_to_vec = |map: &HashMap<OsString, OsString>| -> Vec<_> {
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    };

    let iter = clone_to_vec(&*get_env_store().lock().unwrap()).into_iter();
    Env { iter }
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    get_env_store().lock().unwrap().get(k).cloned()
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let (k, v) = (k.to_owned(), v.to_owned());
    get_env_store().lock().unwrap().insert(k, v);
    Ok(())
}

pub unsafe fn unsetenv(k: &OsStr) -> io::Result<()> {
    get_env_store().lock().unwrap().remove(k);
    Ok(())
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
