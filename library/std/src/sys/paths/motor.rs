use crate::ffi::{OsStr, OsString};
use crate::os::motor::ffi::OsStrExt;
use crate::path::{self, PathBuf};
use crate::sys::pal::map_motor_error;
use crate::{fmt, io, iter, str};

const PATH_SEPARATOR: char = ':';

pub type SplitPaths<'a> = iter::Map<str::Split<'a, char>, fn(&str) -> PathBuf>;

pub fn split_paths(unparsed: &OsStr) -> SplitPaths<'_> {
    fn into_pathbuf(part: &str) -> PathBuf {
        PathBuf::from(part)
    }
    unparsed.as_str().split(PATH_SEPARATOR).map(into_pathbuf as fn(&str) -> PathBuf)
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    let mut joined = String::new();
    for (i, path) in paths.enumerate() {
        let path = path.as_ref().as_str();
        if i > 0 {
            joined.push(PATH_SEPARATOR);
        }
        if path.contains(PATH_SEPARATOR) {
            return Err(JoinPathsError);
        }
        joined.push_str(path);
    }
    Ok(OsString::from(joined))
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "path segment contains separator `{PATH_SEPARATOR}`")
    }
}

impl crate::error::Error for JoinPathsError {}

pub fn getcwd() -> io::Result<PathBuf> {
    moto_rt::fs::getcwd().map(PathBuf::from).map_err(map_motor_error)
}

pub fn chdir(path: &path::Path) -> io::Result<()> {
    moto_rt::fs::chdir(path.as_os_str().as_str()).map_err(map_motor_error)
}

pub fn home_dir() -> Option<PathBuf> {
    Some(PathBuf::from("/user"))
}

pub fn current_exe() -> io::Result<PathBuf> {
    moto_rt::process::current_exe().map(PathBuf::from).map_err(map_motor_error)
}

pub fn temp_dir() -> PathBuf {
    crate::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/user/tmp"))
}
