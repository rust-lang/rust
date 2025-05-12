use crate::ffi::{OsStr, OsString};
use crate::os::windows::prelude::*;
use crate::sys::pal::{c, cvt, fill_utf16_buf, to_u16s};
use crate::{fmt, io, ptr, slice};

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

pub fn getenv(k: &OsStr) -> Option<OsString> {
    let k = to_u16s(k).ok()?;
    fill_utf16_buf(
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
