use crate::collections::HashMap;
use crate::error::Error as StdError;
use crate::ffi::{CStr, OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::os::hermit::ffi::OsStringExt;
use crate::path::{self, PathBuf};
use crate::str;
use crate::sync::Mutex;
use crate::sys::hermit::abi;
use crate::sys::memchr;
use crate::sys::unsupported;
use crate::vec;

pub fn errno() -> i32 {
    0
}

pub fn error_string(_errno: i32) -> String {
    "operation successful".to_string()
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

static mut ENV: Option<Mutex<HashMap<OsString, OsString>>> = None;

pub fn init_environment(env: *const *const i8) {
    unsafe {
        ENV = Some(Mutex::new(HashMap::new()));

        if env.is_null() {
            return;
        }

        let mut guard = ENV.as_ref().unwrap().lock().unwrap();
        let mut environ = env;
        while !(*environ).is_null() {
            if let Some((key, value)) = parse(CStr::from_ptr(*environ).to_bytes()) {
                guard.insert(key, value);
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
    unsafe {
        let guard = ENV.as_ref().unwrap().lock().unwrap();
        let mut result = Vec::new();

        for (key, value) in guard.iter() {
            result.push((key.clone(), value.clone()));
        }

        return Env { iter: result.into_iter() };
    }
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    unsafe { ENV.as_ref().unwrap().lock().unwrap().get_mut(k).cloned() }
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    unsafe {
        let (k, v) = (k.to_owned(), v.to_owned());
        ENV.as_ref().unwrap().lock().unwrap().insert(k, v);
    }
    Ok(())
}

pub fn unsetenv(k: &OsStr) -> io::Result<()> {
    unsafe {
        ENV.as_ref().unwrap().lock().unwrap().remove(k);
    }
    Ok(())
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on hermit")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    unsafe {
        abi::exit(code);
    }
}

pub fn getpid() -> u32 {
    unsafe { abi::getpid() }
}
