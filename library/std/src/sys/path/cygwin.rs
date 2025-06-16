use crate::ffi::OsString;
use crate::os::unix::ffi::OsStringExt;
use crate::path::{Path, PathBuf};
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::cvt;
use crate::{io, ptr};

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

/// Cygwin allways prefers `/` over `\`, and it always converts all `/` to `\`
/// internally when calling Win32 APIs. Therefore, the server component of path
/// `\\?\UNC\localhost/share` is `localhost/share` on Win32, but `localhost`
/// on Cygwin.
#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

pub use super::windows_prefix::parse_prefix;

pub const MAIN_SEP_STR: &str = "/";
pub const MAIN_SEP: char = '/';

unsafe extern "C" {
    // Doc: https://cygwin.com/cygwin-api/func-cygwin-conv-path.html
    // Src: https://github.com/cygwin/cygwin/blob/718a15ba50e0d01c79800bd658c2477f9a603540/winsup/cygwin/path.cc#L3902
    // Safety:
    // * `what` should be `CCP_WIN_A_TO_POSIX` here
    // * `from` is null-terminated UTF-8 path
    // * `to` is buffer, the buffer size is `size`.
    //
    // Converts a path to an absolute POSIX path, no matter the input is Win32 path or POSIX path.
    fn cygwin_conv_path(
        what: libc::c_uint,
        from: *const libc::c_char,
        to: *mut u8,
        size: libc::size_t,
    ) -> libc::ssize_t;
}

const CCP_WIN_A_TO_POSIX: libc::c_uint = 2;

/// Make a POSIX path absolute.
pub(crate) fn absolute(path: &Path) -> io::Result<PathBuf> {
    run_path_with_cstr(path, &|path| {
        let conv = CCP_WIN_A_TO_POSIX;
        let size = cvt(unsafe { cygwin_conv_path(conv, path.as_ptr(), ptr::null_mut(), 0) })?;
        // If success, size should not be 0.
        debug_assert!(size >= 1);
        let size = size as usize;
        let mut buffer = Vec::with_capacity(size);
        cvt(unsafe { cygwin_conv_path(conv, path.as_ptr(), buffer.as_mut_ptr(), size) })?;
        unsafe {
            buffer.set_len(size - 1);
        }
        Ok(PathBuf::from(OsString::from_vec(buffer)))
    })
    .map(|path| {
        if path.prefix().is_some() {
            return path;
        }

        // From unix.rs
        let mut components = path.components();
        let path_os = path.as_os_str().as_encoded_bytes();

        let mut normalized = if path_os.starts_with(b"//") && !path_os.starts_with(b"///") {
            components.next();
            PathBuf::from("//")
        } else {
            PathBuf::new()
        };
        normalized.extend(components);

        if path_os.ends_with(b"/") {
            normalized.push("");
        }

        normalized
    })
}

pub(crate) fn is_absolute(path: &Path) -> bool {
    if path.as_os_str().as_encoded_bytes().starts_with(b"\\") {
        path.has_root() && path.prefix().is_some()
    } else {
        path.has_root()
    }
}
