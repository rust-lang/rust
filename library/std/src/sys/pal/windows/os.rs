//! Implementation of `std::os` functionality for Windows.

#![allow(nonstandard_style)]

#[cfg(test)]
mod tests;

use super::api::{self, WinError};
use super::to_u16s;
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::os::windows::ffi::EncodeWide;
use crate::os::windows::prelude::*;
use crate::path::{self, PathBuf};
use crate::sys::{c, cvt};
use crate::{fmt, io, ptr, slice};

pub fn errno() -> i32 {
    api::get_last_error().code as i32
}

/// Gets a detailed string description for the given error number.
pub fn error_string(mut errnum: i32) -> String {
    let mut buf = [0 as c::WCHAR; 2048];

    unsafe {
        let mut module = ptr::null_mut();
        let mut flags = 0;

        // NTSTATUS errors may be encoded as HRESULT, which may returned from
        // GetLastError. For more information about Windows error codes, see
        // `[MS-ERREF]`: https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/0642cb2f-2075-4469-918c-4441e69c548a
        if (errnum & c::FACILITY_NT_BIT as i32) != 0 {
            // format according to https://support.microsoft.com/en-us/help/259693
            const NTDLL_DLL: &[u16] = &[
                'N' as _, 'T' as _, 'D' as _, 'L' as _, 'L' as _, '.' as _, 'D' as _, 'L' as _,
                'L' as _, 0,
            ];
            module = c::GetModuleHandleW(NTDLL_DLL.as_ptr());

            if !module.is_null() {
                errnum ^= c::FACILITY_NT_BIT as i32;
                flags = c::FORMAT_MESSAGE_FROM_HMODULE;
            }
        }

        let res = c::FormatMessageW(
            flags | c::FORMAT_MESSAGE_FROM_SYSTEM | c::FORMAT_MESSAGE_IGNORE_INSERTS,
            module,
            errnum as u32,
            0,
            buf.as_mut_ptr(),
            buf.len() as u32,
            ptr::null(),
        ) as usize;
        if res == 0 {
            // Sometimes FormatMessageW can fail e.g., system doesn't like 0 as langId,
            let fm_err = errno();
            return format!("OS Error {errnum} (FormatMessageW() returned error {fm_err})");
        }

        match String::from_utf16(&buf[..res]) {
            Ok(mut msg) => {
                // Trim trailing CRLF inserted by FormatMessageW
                let len = msg.trim_end().len();
                msg.truncate(len);
                msg
            }
            Err(..) => format!(
                "OS Error {} (FormatMessageW() returned \
                 invalid UTF-16)",
                errnum
            ),
        }
    }
}

pub struct Env {
    base: *mut c::WCHAR,
    iter: EnvIterator,
}

// FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
pub struct EnvStrDebug<'a> {
    iter: &'a EnvIterator,
}

impl fmt::Debug for EnvStrDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { iter } = self;
        let iter: EnvIterator = (*iter).clone();
        let mut list = f.debug_list();
        for (a, b) in iter {
            list.entry(&(a.to_str().unwrap(), b.to_str().unwrap()));
        }
        list.finish()
    }
}

impl Env {
    pub fn str_debug(&self) -> impl fmt::Debug + '_ {
        let Self { base: _, iter } = self;
        EnvStrDebug { iter }
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { base: _, iter } = self;
        f.debug_list().entries(iter.clone()).finish()
    }
}

impl Iterator for Env {
    type Item = (OsString, OsString);

    fn next(&mut self) -> Option<(OsString, OsString)> {
        let Self { base: _, iter } = self;
        iter.next()
    }
}

#[derive(Clone)]
struct EnvIterator(*mut c::WCHAR);

impl Iterator for EnvIterator {
    type Item = (OsString, OsString);

    fn next(&mut self) -> Option<(OsString, OsString)> {
        let Self(cur) = self;
        loop {
            unsafe {
                if **cur == 0 {
                    return None;
                }
                let p = *cur as *const u16;
                let mut len = 0;
                while *p.add(len) != 0 {
                    len += 1;
                }
                let s = slice::from_raw_parts(p, len);
                *cur = cur.add(len + 1);

                // Windows allows environment variables to start with an equals
                // symbol (in any other position, this is the separator between
                // variable name and value). Since`s` has at least length 1 at
                // this point (because the empty string terminates the array of
                // environment variables), we can safely slice.
                let pos = match s[1..].iter().position(|&u| u == b'=' as u16).map(|p| p + 1) {
                    Some(p) => p,
                    None => continue,
                };
                return Some((
                    OsStringExt::from_wide(&s[..pos]),
                    OsStringExt::from_wide(&s[pos + 1..]),
                ));
            }
        }
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        unsafe {
            c::FreeEnvironmentStringsW(self.base);
        }
    }
}

pub fn env() -> Env {
    unsafe {
        let ch = c::GetEnvironmentStringsW();
        if ch.is_null() {
            panic!("failure getting env string from OS: {}", io::Error::last_os_error());
        }
        Env { base: ch, iter: EnvIterator(ch) }
    }
}

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

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "failed to join paths"
    }
}

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

pub fn getenv(k: &OsStr) -> Option<OsString> {
    let k = to_u16s(k).ok()?;
    super::fill_utf16_buf(
        |buf, sz| unsafe { c::GetEnvironmentVariableW(k.as_ptr(), buf, sz) },
        OsStringExt::from_wide,
    )
    .ok()
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    // SAFETY: We ensure that k and v are null-terminated wide strings.
    unsafe {
        let k = to_u16s(k)?;
        let v = to_u16s(v)?;

        cvt(c::SetEnvironmentVariableW(k.as_ptr(), v.as_ptr())).map(drop)
    }
}

pub unsafe fn unsetenv(n: &OsStr) -> io::Result<()> {
    // SAFETY: We ensure that v is a null-terminated wide strings.
    unsafe {
        let v = to_u16s(n)?;
        cvt(c::SetEnvironmentVariableW(v.as_ptr(), ptr::null())).map(drop)
    }
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
