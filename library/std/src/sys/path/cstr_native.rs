use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::path::{AsPath, Path};
use crate::sys::common::small_c_string::run_path_with_cstr;

pub type NativePath = CStr;

#[unstable(feature = "fs_native_path_internals", issue = "none")]
impl<P: AsRef<Path>> AsPath for P {
    #[doc(hidden)]
    fn with_path<T, F: FnOnce(&Path) -> io::Result<T>>(self, f: F) -> io::Result<T> {
        f(self.as_ref())
    }
    #[doc(hidden)]
    fn with_native_path<T, F: Fn(&NativePath) -> io::Result<T>>(self, f: F) -> io::Result<T> {
        run_path_with_cstr(self.as_ref(), &f)
    }
}

#[unstable(feature = "fs_native_path_internals", issue = "none")]
impl AsPath for &crate::path::NativePath {
    #[doc(hidden)]
    fn with_path<T, F: FnOnce(&Path) -> io::Result<T>>(self, f: F) -> io::Result<T> {
        // SAFETY: OsStr is a byte slice on platforms with CStr paths.
        // Note: We can't use os::unix::OsStrExt because that isn't necessarily
        // available for all platforms that use CStr paths.
        let osstr: &Path = unsafe { mem::transmute(self.0.to_bytes()) };
        f(osstr)
    }
    #[doc(hidden)]
    fn with_native_path<T, F: Fn(&NativePath) -> io::Result<T>>(self, f: F) -> io::Result<T> {
        f(&self.0)
    }
}
