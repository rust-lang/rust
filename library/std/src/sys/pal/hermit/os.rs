use core::slice::memchr;

use super::hermit_abi;
use crate::collections::HashMap;
use crate::error::Error as StdError;
use crate::ffi::{CStr, OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::hermit::ffi::OsStringExt;
use crate::path::{self, PathBuf};
use crate::sync::Mutex;
use crate::sys::unsupported;
use crate::{fmt, io, str, vec};

pub fn errno() -> i32 {
    unsafe { hermit_abi::get_errno() }
}

pub fn error_string(errno: i32) -> String {
    hermit_abi::error_string(errno).to_string()
}

pub fn getcwd() -> io::Result<PathBuf> {
    Ok(PathBuf::from("/"))
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
        "not supported on hermit yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "not supported on hermit yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

static ENV: Mutex<Option<HashMap<OsString, OsString>>> = Mutex::new(None);

pub fn init_environment(env: *const *const i8) {
    let mut guard = ENV.lock().unwrap();
    let map = guard.insert(HashMap::new());

    if env.is_null() {
        return;
    }

    unsafe {
        let mut environ = env;
        while !(*environ).is_null() {
            if let Some((key, value)) = parse(CStr::from_ptr(*environ).to_bytes()) {
                map.insert(key, value);
            }
            environ = environ.add(1);
        }
    }

    fn parse(input: &[u8]) -> Option<(OsString, OsString)> {
        // Strategy (copied from glibc): Variable name and value are separated
        // by an ASCII equals sign '='. Since a variable name must not be
        // empty, allow variable names starting with an equals sign. Skip all
        // malformed lines.
        if input.is_empty() {
            return None;
        }
        let pos = memchr::memchr(b'=', &input[1..]).map(|p| p + 1);
        pos.map(|p| {
            (
                OsStringExt::from_vec(input[..p].to_vec()),
                OsStringExt::from_vec(input[p + 1..].to_vec()),
            )
        })
    }
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

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env() -> Env {
    let guard = ENV.lock().unwrap();
    let env = guard.as_ref().unwrap();

    let result = env.iter().map(|(key, value)| (key.clone(), value.clone())).collect::<Vec<_>>();

    Env { iter: result.into_iter() }
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    ENV.lock().unwrap().as_ref().unwrap().get(k).cloned()
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let (k, v) = (k.to_owned(), v.to_owned());
    ENV.lock().unwrap().as_mut().unwrap().insert(k, v);
    Ok(())
}

pub unsafe fn unsetenv(k: &OsStr) -> io::Result<()> {
    ENV.lock().unwrap().as_mut().unwrap().remove(k);
    Ok(())
}

pub fn temp_dir() -> PathBuf {
    PathBuf::from("/tmp")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    unsafe { hermit_abi::exit(code) }
}

pub fn getpid() -> u32 {
    unsafe { hermit_abi::getpid() as u32 }
}
