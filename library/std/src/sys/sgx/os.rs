use fortanix_sgx_abi::{Error, RESULT_SUCCESS};

use crate::collections::HashMap;
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::path::{self, PathBuf};
use crate::str;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::Mutex;
use crate::sync::Once;
use crate::sys::{decode_error_kind, sgx_ineffective, unsupported};
use crate::vec;

pub fn errno() -> i32 {
    RESULT_SUCCESS
}

pub fn error_string(errno: i32) -> String {
    if errno == RESULT_SUCCESS {
        "operation successful".into()
    } else if ((Error::UserRangeStart as _)..=(Error::UserRangeEnd as _)).contains(&errno) {
        format!("user-specified error {:08x}", errno)
    } else {
        decode_error_kind(errno).as_str().into()
    }
}

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    sgx_ineffective(())
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
        "not supported in SGX yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "not supported in SGX yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

#[cfg_attr(test, linkage = "available_externally")]
#[export_name = "_ZN16__rust_internals3std3sys3sgx2os3ENVE"]
static ENV: AtomicUsize = AtomicUsize::new(0);
#[cfg_attr(test, linkage = "available_externally")]
#[export_name = "_ZN16__rust_internals3std3sys3sgx2os8ENV_INITE"]
static ENV_INIT: Once = Once::new();
type EnvStore = Mutex<HashMap<OsString, OsString>>;

fn get_env_store() -> Option<&'static EnvStore> {
    unsafe { (ENV.load(Ordering::Relaxed) as *const EnvStore).as_ref() }
}

fn create_env_store() -> &'static EnvStore {
    ENV_INIT.call_once(|| {
        ENV.store(Box::into_raw(Box::new(EnvStore::default())) as _, Ordering::Relaxed)
    });
    unsafe { &*(ENV.load(Ordering::Relaxed) as *const EnvStore) }
}

pub type Env = vec::IntoIter<(OsString, OsString)>;

pub fn env() -> Env {
    let clone_to_vec = |map: &HashMap<OsString, OsString>| -> Vec<_> {
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    };

    get_env_store().map(|env| clone_to_vec(&env.lock().unwrap())).unwrap_or_default().into_iter()
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    get_env_store().and_then(|s| s.lock().unwrap().get(k).cloned())
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let (k, v) = (k.to_owned(), v.to_owned());
    create_env_store().lock().unwrap().insert(k, v);
    Ok(())
}

pub fn unsetenv(k: &OsStr) -> io::Result<()> {
    if let Some(env) = get_env_store() {
        env.lock().unwrap().remove(k);
    }
    Ok(())
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem in SGX")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    super::abi::exit_with_code(code as _)
}

pub fn getpid() -> u32 {
    panic!("no pids in SGX")
}
