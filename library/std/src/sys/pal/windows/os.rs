//! Implementation of `std::os` functionality for Windows.

#![allow(nonstandard_style)]

#[cfg(test)]
mod tests;

use super::api;
#[cfg(not(target_vendor = "uwp"))]
use super::api::WinError;
use crate::ffi::{OsStr, OsString};
use crate::os::windows::ffi::EncodeWide;
use crate::os::windows::prelude::*;
use crate::path::{self, PathBuf};
use crate::sys::pal::{c, cvt};
use crate::{fmt, io, ptr};

pub struct SplitPaths<'a> {
    data: EncodeWide<'a>,
    must_yield: bool,
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths<'_> {
    SplitPaths { data: unparsed.encode_wide(), must_yield: true }
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        // On Windows, the PATH environment variable is semicolon separated.
        // Double quotes are used as a way of introducing literal semicolons
        // (since c:\some;dir is a valid Windows path). Double quotes are not
        // themselves permitted in path names, so there is no way to escape a
        // double quote. Quoted regions can appear in arbitrary locations, so
        //
        //   c:\foo;c:\som"e;di"r;c:\bar
        //
        // Should parse as [c:\foo, c:\some;dir, c:\bar].
        //
        // (The above is based on testing; there is no clear reference available
        // for the grammar.)

        let must_yield = self.must_yield;
        self.must_yield = false;

        let mut in_progress = Vec::new();
        let mut in_quote = false;
        for b in self.data.by_ref() {
            if b == '"' as u16 {
                in_quote = !in_quote;
            } else if b == ';' as u16 && !in_quote {
                self.must_yield = true;
                break;
            } else {
                in_progress.push(b)
            }
        }

        if !must_yield && in_progress.is_empty() {
            None
        } else {
            Some(super::os2path(&in_progress))
        }
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    let mut joined = Vec::new();
    let sep = b';' as u16;

    for (i, path) in paths.enumerate() {
        let path = path.as_ref();
        if i > 0 {
            joined.push(sep)
        }
        let v = path.encode_wide().collect::<Vec<u16>>();
        if v.contains(&(b'"' as u16)) {
            return Err(JoinPathsError);
        } else if v.contains(&sep) {
            joined.push(b'"' as u16);
            joined.extend_from_slice(&v[..]);
            joined.push(b'"' as u16);
        } else {
            joined.extend_from_slice(&v[..]);
        }
    }

    Ok(OsStringExt::from_wide(&joined[..]))
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "path segment contains `\"`".fmt(f)
    }
}

impl crate::error::Error for JoinPathsError {}

pub fn current_exe() -> io::Result<PathBuf> {
    super::fill_utf16_buf(
        |buf, sz| unsafe { c::GetModuleFileNameW(ptr::null_mut(), buf, sz) },
        super::os2path,
    )
}

pub fn getcwd() -> io::Result<PathBuf> {
    super::fill_utf16_buf(|buf, sz| unsafe { c::GetCurrentDirectoryW(sz, buf) }, super::os2path)
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    let p: &OsStr = p.as_ref();
    let mut p = p.encode_wide().collect::<Vec<_>>();
    p.push(0);

    cvt(unsafe { c::SetCurrentDirectoryW(p.as_ptr()) }).map(drop)
}

pub fn temp_dir() -> PathBuf {
    super::fill_utf16_buf(|buf, sz| unsafe { c::GetTempPath2W(sz, buf) }, super::os2path).unwrap()
}

#[cfg(all(not(target_vendor = "uwp"), not(target_vendor = "win7")))]
fn home_dir_crt() -> Option<PathBuf> {
    unsafe {
        // Defined in processthreadsapi.h.
        const CURRENT_PROCESS_TOKEN: usize = -4_isize as usize;

        super::fill_utf16_buf(
            |buf, mut sz| {
                // GetUserProfileDirectoryW does not quite use the usual protocol for
                // negotiating the buffer size, so we have to translate.
                match c::GetUserProfileDirectoryW(
                    ptr::without_provenance_mut(CURRENT_PROCESS_TOKEN),
                    buf,
                    &mut sz,
                ) {
                    0 if api::get_last_error() != WinError::INSUFFICIENT_BUFFER => 0,
                    0 => sz,
                    _ => sz - 1, // sz includes the null terminator
                }
            },
            super::os2path,
        )
        .ok()
    }
}

#[cfg(target_vendor = "win7")]
fn home_dir_crt() -> Option<PathBuf> {
    unsafe {
        use crate::sys::handle::Handle;

        let me = c::GetCurrentProcess();
        let mut token = ptr::null_mut();
        if c::OpenProcessToken(me, c::TOKEN_READ, &mut token) == 0 {
            return None;
        }
        let _handle = Handle::from_raw_handle(token);
        super::fill_utf16_buf(
            |buf, mut sz| {
                match c::GetUserProfileDirectoryW(token, buf, &mut sz) {
                    0 if api::get_last_error() != WinError::INSUFFICIENT_BUFFER => 0,
                    0 => sz,
                    _ => sz - 1, // sz includes the null terminator
                }
            },
            super::os2path,
        )
        .ok()
    }
}

#[cfg(target_vendor = "uwp")]
fn home_dir_crt() -> Option<PathBuf> {
    None
}

pub fn home_dir() -> Option<PathBuf> {
    crate::env::var_os("USERPROFILE")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(home_dir_crt)
}

pub fn exit(code: i32) -> ! {
    unsafe { c::ExitProcess(code as u32) }
}

pub fn getpid() -> u32 {
    unsafe { c::GetCurrentProcessId() }
}
