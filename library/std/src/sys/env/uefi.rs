use crate::ffi::{OsStr, OsString};
use crate::{fmt, io};

pub struct EnvStrDebug<'a> {
    iter: &'a [(OsString, OsString)],
}

impl fmt::Debug for EnvStrDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for (a, b) in self.iter {
            list.entry(&(a.to_str().unwrap(), b.to_str().unwrap()));
        }
        list.finish()
    }
}

pub struct Env(crate::vec::IntoIter<(OsString, OsString)>);

impl Env {
    // FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
    pub fn str_debug(&self) -> impl fmt::Debug + '_ {
        EnvStrDebug { iter: self.0.as_slice() }
    }
}

impl Iterator for Env {
    type Item = (OsString, OsString);

    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.0.next()
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub fn env() -> Env {
    let env = uefi_env::get_all().expect("not supported on this platform");
    Env(env.into_iter())
}

pub fn getenv(key: &OsStr) -> Option<OsString> {
    uefi_env::get(key)
}

pub unsafe fn setenv(key: &OsStr, val: &OsStr) -> io::Result<()> {
    uefi_env::set(key, val)
}

pub unsafe fn unsetenv(key: &OsStr) -> io::Result<()> {
    uefi_env::unset(key)
}

mod uefi_env {
    use crate::ffi::{OsStr, OsString};
    use crate::io;
    use crate::os::uefi::ffi::OsStringExt;
    use crate::ptr::NonNull;
    use crate::sys::{helpers, unsupported_err};

    pub(crate) fn get(key: &OsStr) -> Option<OsString> {
        let shell = helpers::open_shell()?;
        let mut key_ptr = helpers::os_string_to_raw(key)?;
        unsafe { get_raw(shell, key_ptr.as_mut_ptr()) }
    }

    pub(crate) fn set(key: &OsStr, val: &OsStr) -> io::Result<()> {
        let mut key_ptr = helpers::os_string_to_raw(key)
            .ok_or(io::const_error!(io::ErrorKind::InvalidInput, "invalid key"))?;
        let mut val_ptr = helpers::os_string_to_raw(val)
            .ok_or(io::const_error!(io::ErrorKind::InvalidInput, "invalid value"))?;
        unsafe { set_raw(key_ptr.as_mut_ptr(), val_ptr.as_mut_ptr()) }
    }

    pub(crate) fn unset(key: &OsStr) -> io::Result<()> {
        let mut key_ptr = helpers::os_string_to_raw(key)
            .ok_or(io::const_error!(io::ErrorKind::InvalidInput, "invalid key"))?;
        unsafe { set_raw(key_ptr.as_mut_ptr(), crate::ptr::null_mut()) }
    }

    pub(crate) fn get_all() -> io::Result<Vec<(OsString, OsString)>> {
        let shell = helpers::open_shell().ok_or(unsupported_err())?;

        let mut vars = Vec::new();
        let val = unsafe { ((*shell.as_ptr()).get_env)(crate::ptr::null_mut()) };

        if val.is_null() {
            return Ok(vars);
        }

        let mut start = 0;

        // UEFI Shell returns all keys separated by NULL.
        // End of string is denoted by two NULLs
        for i in 0.. {
            if unsafe { *val.add(i) } == 0 {
                // Two NULL signal end of string
                if i == start {
                    break;
                }

                let key = OsString::from_wide(unsafe {
                    crate::slice::from_raw_parts(val.add(start), i - start)
                });
                // SAFETY: val.add(start) is always NULL terminated
                let val = unsafe { get_raw(shell, val.add(start)) }
                    .ok_or(io::const_error!(io::ErrorKind::InvalidInput, "invalid value"))?;

                vars.push((key, val));
                start = i + 1;
            }
        }

        Ok(vars)
    }

    unsafe fn get_raw(
        shell: NonNull<r_efi::efi::protocols::shell::Protocol>,
        key_ptr: *mut r_efi::efi::Char16,
    ) -> Option<OsString> {
        let val = unsafe { ((*shell.as_ptr()).get_env)(key_ptr) };
        helpers::os_string_from_raw(val)
    }

    unsafe fn set_raw(
        key_ptr: *mut r_efi::efi::Char16,
        val_ptr: *mut r_efi::efi::Char16,
    ) -> io::Result<()> {
        let shell = helpers::open_shell().ok_or(unsupported_err())?;
        let r =
            unsafe { ((*shell.as_ptr()).set_env)(key_ptr, val_ptr, r_efi::efi::Boolean::FALSE) };
        if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
    }
}
