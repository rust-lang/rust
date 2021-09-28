use super::unsupported;
use crate::error::Error as StdError;
use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::os::{
    raw::{c_char, c_int},
    solid::ffi::{OsStrExt, OsStringExt},
};
use crate::path::{self, PathBuf};
use crate::sys_common::rwlock::StaticRWLock;
use crate::vec;

use super::{abi, error, itron, memchr};

// `solid` directly maps `errno`s to μITRON error codes.
impl itron::error::ItronError {
    #[inline]
    pub(crate) fn as_io_error(self) -> crate::io::Error {
        crate::io::Error::from_raw_os_error(self.as_raw())
    }
}

pub fn errno() -> i32 {
    0
}

pub fn error_string(errno: i32) -> String {
    if let Some(name) = error::error_name(errno) { name.to_owned() } else { format!("{}", errno) }
}

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    unsupported()
}

pub struct SplitPaths<'a>(&'a !);

pub fn split_paths(_unparsed: &OsStr) -> SplitPaths<'_> {
    panic!("unsupported")
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        *self.0
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

static ENV_LOCK: StaticRWLock = StaticRWLock::new();

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
    extern "C" {
        static mut environ: *const *const c_char;
    }

    unsafe {
        let _guard = ENV_LOCK.read();
        let mut result = Vec::new();
        if !environ.is_null() {
            while !(*environ).is_null() {
                if let Some(key_value) = parse(CStr::from_ptr(*environ).to_bytes()) {
                    result.push(key_value);
                }
                environ = environ.add(1);
            }
        }
        return Env { iter: result.into_iter() };
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

pub fn getenv(k: &OsStr) -> Option<OsString> {
    // environment variables with a nul byte can't be set, so their value is
    // always None as well
    let k = CString::new(k.as_bytes()).ok()?;
    unsafe {
        let _guard = ENV_LOCK.read();
        let s = libc::getenv(k.as_ptr()) as *const libc::c_char;
        if s.is_null() {
            None
        } else {
            Some(OsStringExt::from_vec(CStr::from_ptr(s).to_bytes().to_vec()))
        }
    }
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let k = CString::new(k.as_bytes())?;
    let v = CString::new(v.as_bytes())?;

    unsafe {
        let _guard = ENV_LOCK.write();
        cvt_env(libc::setenv(k.as_ptr(), v.as_ptr(), 1)).map(drop)
    }
}

pub fn unsetenv(n: &OsStr) -> io::Result<()> {
    let nbuf = CString::new(n.as_bytes())?;

    unsafe {
        let _guard = ENV_LOCK.write();
        cvt_env(libc::unsetenv(nbuf.as_ptr())).map(drop)
    }
}

/// In kmclib, `setenv` and `unsetenv` don't always set `errno`, so this
/// function just returns a generic error.
fn cvt_env(t: c_int) -> io::Result<c_int> {
    if t == -1 {
        Err(io::Error::new_const(io::ErrorKind::Uncategorized, &"failure"))
    } else {
        Ok(t)
    }
}

pub fn temp_dir() -> PathBuf {
    panic!("no standard temporary directory on this platform")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(_code: i32) -> ! {
    let tid = itron::task::try_current_task_id().unwrap_or(0);
    loop {
        abi::breakpoint_program_exited(tid as usize);
    }
}

pub fn getpid() -> u32 {
    panic!("no pids on this platform")
}
