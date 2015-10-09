use sys::inner::prelude::*;
use sys::os_str::prelude as os_str;
use sys::error::prelude as error;

use ffi::{OsStr, OsString};
use path::{Path, PathBuf};
use io;

struct Conv<T>(T);

impl<T, U> Into<io::Result<U>> for Conv<Result<T, error::Error>> where Conv<T>: Into<U> {
    fn into(self) -> io::Result<U> {
        self.0.map_err(From::from).map(conv)
    }
}

impl Into<io::Error> for Conv<error::Error> {
    fn into(self) -> io::Error {
        From::from(self.0)
    }
}

impl Into<PathBuf> for Conv<os_str::OsString> {
    fn into(self) -> PathBuf {
        PathBuf::from(OsString::from_inner(self.0))
    }
}

impl Into<OsString> for Conv<os_str::OsString> {
    fn into(self) -> OsString {
        OsString::from_inner(self.0)
    }
}

impl<'a> Into<&'a os_str::OsStr> for Conv<&'a OsStr> {
    fn into(self) -> &'a os_str::OsStr {
        self.0.as_inner()
    }
}

impl<'a> Into<&'a os_str::OsStr> for Conv<&'a Path> {
    fn into(self) -> &'a os_str::OsStr {
        self.0.as_os_str().as_inner()
    }
}

impl<'a> Into<&'a OsStr> for Conv<&'a os_str::OsStr> {
    fn into(self) -> &'a OsStr {
        FromInner::from_inner(self.0)
    }
}

impl Into<()> for Conv<()> {
    fn into(self) -> () {
        self.0
    }
}

impl Into<usize> for Conv<usize> {
    fn into(self) -> usize {
        self.0
    }
}

impl Into<u64> for Conv<u64> {
    fn into(self) -> u64 {
        self.0
    }
}

impl<T: AsRef<U>, U> AsRef<U> for Conv<T> {
    fn as_ref(&self) -> &U {
        self.0.as_ref()
    }
}

pub fn conv<T, U>(t: T) -> U where Conv<T>: Into<U> {
    Conv(t).into()
}
