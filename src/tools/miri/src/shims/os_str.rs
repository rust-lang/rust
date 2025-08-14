use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(windows)]
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use rustc_middle::ty::Ty;

use crate::*;

/// Represent how path separator conversion should be done.
pub enum PathConversion {
    HostToTarget,
    TargetToHost,
}

#[cfg(unix)]
pub fn bytes_to_os_str<'tcx>(bytes: &[u8]) -> InterpResult<'tcx, &OsStr> {
    interp_ok(OsStr::from_bytes(bytes))
}
#[cfg(not(unix))]
pub fn bytes_to_os_str<'tcx>(bytes: &[u8]) -> InterpResult<'tcx, &OsStr> {
    // We cannot use `from_encoded_bytes_unchecked` here since we can't trust `bytes`.
    let s = std::str::from_utf8(bytes)
        .map_err(|_| err_unsup_format!("{:?} is not a valid utf-8 string", bytes))?;
    interp_ok(OsStr::new(s))
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Helper function to read an OsString from a null-terminated sequence of bytes, which is what
    /// the Unix APIs usually handle.
    fn read_os_str_from_c_str<'a>(&'a self, ptr: Pointer) -> InterpResult<'tcx, &'a OsStr>
    where
        'tcx: 'a,
    {
        let this = self.eval_context_ref();
        let bytes = this.read_c_str(ptr)?;
        bytes_to_os_str(bytes)
    }

    /// Helper function to read an OsString from a 0x0000-terminated sequence of u16,
    /// which is what the Windows APIs usually handle.
    fn read_os_str_from_wide_str<'a>(&'a self, ptr: Pointer) -> InterpResult<'tcx, OsString>
    where
        'tcx: 'a,
    {
        #[cfg(windows)]
        pub fn u16vec_to_osstring<'tcx>(u16_vec: Vec<u16>) -> InterpResult<'tcx, OsString> {
            interp_ok(OsString::from_wide(&u16_vec[..]))
        }
        #[cfg(not(windows))]
        pub fn u16vec_to_osstring<'tcx>(u16_vec: Vec<u16>) -> InterpResult<'tcx, OsString> {
            let s = String::from_utf16(&u16_vec[..])
                .map_err(|_| err_unsup_format!("{:?} is not a valid utf-16 string", u16_vec))?;
            interp_ok(s.into())
        }

        let u16_vec = self.eval_context_ref().read_wide_str(ptr)?;
        u16vec_to_osstring(u16_vec)
    }

    /// Helper function to write an OsStr as a null-terminated sequence of bytes, which is what the
    /// Unix APIs usually handle. Returns `(success, full_len)`, where length includes the null
    /// terminator. On failure, nothing is written.
    fn write_os_str_to_c_str(
        &mut self,
        os_str: &OsStr,
        ptr: Pointer,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        let bytes = os_str.as_encoded_bytes();
        self.eval_context_mut().write_c_str(bytes, ptr, size)
    }

    /// Internal helper to share code between `write_os_str_to_wide_str` and
    /// `write_os_str_to_wide_str_truncated`.
    fn write_os_str_to_wide_str_helper(
        &mut self,
        os_str: &OsStr,
        ptr: Pointer,
        size: u64,
        truncate: bool,
    ) -> InterpResult<'tcx, (bool, u64)> {
        #[cfg(windows)]
        fn os_str_to_u16vec<'tcx>(os_str: &OsStr) -> InterpResult<'tcx, Vec<u16>> {
            interp_ok(os_str.encode_wide().collect())
        }
        #[cfg(not(windows))]
        fn os_str_to_u16vec<'tcx>(os_str: &OsStr) -> InterpResult<'tcx, Vec<u16>> {
            // On non-Windows platforms the best we can do to transform Vec<u16> from/to OS strings is to do the
            // intermediate transformation into strings. Which invalidates non-utf8 paths that are actually
            // valid.
            os_str
                .to_str()
                .map(|s| s.encode_utf16().collect())
                .ok_or_else(|| err_unsup_format!("{:?} is not a valid utf-8 string", os_str))
                .into()
        }

        let u16_vec = os_str_to_u16vec(os_str)?;
        let (written, size_needed) = self.eval_context_mut().write_wide_str(&u16_vec, ptr, size)?;
        if truncate && !written && size > 0 {
            // Write the truncated part that fits.
            let truncated_data = &u16_vec[..size.saturating_sub(1).try_into().unwrap()];
            let (written, written_len) =
                self.eval_context_mut().write_wide_str(truncated_data, ptr, size)?;
            assert!(written && written_len == size);
        }
        interp_ok((written, size_needed))
    }

    /// Helper function to write an OsStr as a 0x0000-terminated u16-sequence, which is what the
    /// Windows APIs usually handle. Returns `(success, full_len)`, where length is measured
    /// in units of `u16` and includes the null terminator. On failure, nothing is written.
    fn write_os_str_to_wide_str(
        &mut self,
        os_str: &OsStr,
        ptr: Pointer,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        self.write_os_str_to_wide_str_helper(os_str, ptr, size, /*truncate*/ false)
    }

    /// Like `write_os_str_to_wide_str`, but on failure as much as possible is written into
    /// the buffer (always with a null terminator).
    fn write_os_str_to_wide_str_truncated(
        &mut self,
        os_str: &OsStr,
        ptr: Pointer,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        self.write_os_str_to_wide_str_helper(os_str, ptr, size, /*truncate*/ true)
    }

    /// Allocate enough memory to store the given `OsStr` as a null-terminated sequence of bytes.
    fn alloc_os_str_as_c_str(
        &mut self,
        os_str: &OsStr,
        memkind: MemoryKind,
    ) -> InterpResult<'tcx, Pointer> {
        let size = u64::try_from(os_str.len()).unwrap().strict_add(1); // Make space for `0` terminator.
        let this = self.eval_context_mut();

        let arg_type = Ty::new_array(this.tcx.tcx, this.tcx.types.u8, size);
        let arg_place = this.allocate(this.layout_of(arg_type).unwrap(), memkind)?;
        let (written, _) = self.write_os_str_to_c_str(os_str, arg_place.ptr(), size).unwrap();
        assert!(written);
        interp_ok(arg_place.ptr())
    }

    /// Allocate enough memory to store the given `OsStr` as a null-terminated sequence of `u16`.
    fn alloc_os_str_as_wide_str(
        &mut self,
        os_str: &OsStr,
        memkind: MemoryKind,
    ) -> InterpResult<'tcx, Pointer> {
        let size = u64::try_from(os_str.len()).unwrap().strict_add(1); // Make space for `0x0000` terminator.
        let this = self.eval_context_mut();

        let arg_type = Ty::new_array(this.tcx.tcx, this.tcx.types.u16, size);
        let arg_place = this.allocate(this.layout_of(arg_type).unwrap(), memkind)?;
        let (written, _) = self.write_os_str_to_wide_str(os_str, arg_place.ptr(), size).unwrap();
        assert!(written);
        interp_ok(arg_place.ptr())
    }

    /// Read a null-terminated sequence of bytes, and perform path separator conversion if needed.
    fn read_path_from_c_str<'a>(&'a self, ptr: Pointer) -> InterpResult<'tcx, Cow<'a, Path>>
    where
        'tcx: 'a,
    {
        let this = self.eval_context_ref();
        let os_str = this.read_os_str_from_c_str(ptr)?;

        interp_ok(match this.convert_path(Cow::Borrowed(os_str), PathConversion::TargetToHost) {
            Cow::Borrowed(x) => Cow::Borrowed(Path::new(x)),
            Cow::Owned(y) => Cow::Owned(PathBuf::from(y)),
        })
    }

    /// Read a null-terminated sequence of `u16`s, and perform path separator conversion if needed.
    fn read_path_from_wide_str(&self, ptr: Pointer) -> InterpResult<'tcx, PathBuf> {
        let this = self.eval_context_ref();
        let os_str = this.read_os_str_from_wide_str(ptr)?;

        interp_ok(
            this.convert_path(Cow::Owned(os_str), PathConversion::TargetToHost).into_owned().into(),
        )
    }

    /// Write a Path to the machine memory (as a null-terminated sequence of bytes),
    /// adjusting path separators if needed.
    fn write_path_to_c_str(
        &mut self,
        path: &Path,
        ptr: Pointer,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        let this = self.eval_context_mut();
        let os_str =
            this.convert_path(Cow::Borrowed(path.as_os_str()), PathConversion::HostToTarget);
        this.write_os_str_to_c_str(&os_str, ptr, size)
    }

    /// Write a Path to the machine memory (as a null-terminated sequence of `u16`s),
    /// adjusting path separators if needed.
    fn write_path_to_wide_str(
        &mut self,
        path: &Path,
        ptr: Pointer,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        let this = self.eval_context_mut();
        let os_str =
            this.convert_path(Cow::Borrowed(path.as_os_str()), PathConversion::HostToTarget);
        this.write_os_str_to_wide_str(&os_str, ptr, size)
    }

    /// Write a Path to the machine memory (as a null-terminated sequence of `u16`s),
    /// adjusting path separators if needed.
    fn write_path_to_wide_str_truncated(
        &mut self,
        path: &Path,
        ptr: Pointer,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        let this = self.eval_context_mut();
        let os_str =
            this.convert_path(Cow::Borrowed(path.as_os_str()), PathConversion::HostToTarget);
        this.write_os_str_to_wide_str_truncated(&os_str, ptr, size)
    }

    /// Allocate enough memory to store a Path as a null-terminated sequence of bytes,
    /// adjusting path separators if needed.
    fn alloc_path_as_c_str(
        &mut self,
        path: &Path,
        memkind: MemoryKind,
    ) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();
        let os_str =
            this.convert_path(Cow::Borrowed(path.as_os_str()), PathConversion::HostToTarget);
        this.alloc_os_str_as_c_str(&os_str, memkind)
    }

    /// Allocate enough memory to store a Path as a null-terminated sequence of `u16`s,
    /// adjusting path separators if needed.
    fn alloc_path_as_wide_str(
        &mut self,
        path: &Path,
        memkind: MemoryKind,
    ) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();
        let os_str =
            this.convert_path(Cow::Borrowed(path.as_os_str()), PathConversion::HostToTarget);
        this.alloc_os_str_as_wide_str(&os_str, memkind)
    }

    fn convert_path<'a>(
        &self,
        os_str: Cow<'a, OsStr>,
        direction: PathConversion,
    ) -> Cow<'a, OsStr> {
        let this = self.eval_context_ref();
        let target_os = &this.tcx.sess.target.os;

        /// Adjust a Windows path to Unix conventions such that it un-does everything that
        /// `unix_to_windows` did, and such that if the Windows input path was absolute, then the
        /// Unix output path is absolute.
        fn windows_to_unix<T>(path: &mut Vec<T>)
        where
            T: From<u8> + Copy + Eq,
        {
            let sep = T::from(b'/');
            // Make sure all path separators are `/`.
            for c in path.iter_mut() {
                if *c == b'\\'.into() {
                    *c = sep;
                }
            }
            // If this starts with `//?/`, it was probably produced by `unix_to_windows`` and we
            // remove the `//?` that got added to get the Unix path back out.
            if path.get(0..4) == Some(&[sep, sep, b'?'.into(), sep]) {
                // Remove first 3 characters. It still starts with `/` so it is absolute on Unix.
                path.splice(0..3, std::iter::empty());
            }
            // If it starts with a drive letter (`X:/`), convert it to an absolute Unix path.
            else if path.get(1..3) == Some(&[b':'.into(), sep]) {
                // We add a `/` at the beginning, to store the absolute Windows
                // path in something that looks like an absolute Unix path.
                path.insert(0, sep);
            }
        }

        /// Adjust a Unix path to Windows conventions such that it un-does everything that
        /// `windows_to_unix` did, and such that if the Unix input path was absolute, then the
        /// Windows output path is absolute.
        fn unix_to_windows<T>(path: &mut Vec<T>)
        where
            T: From<u8> + Copy + Eq,
        {
            let sep = T::from(b'\\');
            // Make sure all path separators are `\`.
            for c in path.iter_mut() {
                if *c == b'/'.into() {
                    *c = sep;
                }
            }
            // If the path is `\X:\`, the leading separator was probably added by `windows_to_unix`
            // and we should get rid of it again.
            if path.get(2..4) == Some(&[b':'.into(), sep]) && path[0] == sep {
                // The new path is still absolute on Windows.
                path.remove(0);
            }
            // If this starts withs a `\` but not a `\\`, then this was absolute on Unix but is
            // relative on Windows (relative to "the root of the current directory", e.g. the
            // drive letter).
            else if path.first() == Some(&sep) && path.get(1) != Some(&sep) {
                // We add `\\?` so it starts with `\\?\` which is some magic path on Windows
                // that *is* considered absolute. This way we store the absolute Unix path
                // in something that looks like an absolute Windows path.
                path.splice(0..0, [sep, sep, b'?'.into()]);
            }
        }

        // Below we assume that everything non-Windows works like Unix, at least
        // when it comes to file system path conventions.
        #[cfg(windows)]
        return if target_os == "windows" {
            // Windows-on-Windows, all fine.
            os_str
        } else {
            // Unix target, Windows host.
            let mut path: Vec<u16> = os_str.encode_wide().collect();
            match direction {
                PathConversion::HostToTarget => {
                    windows_to_unix(&mut path);
                }
                PathConversion::TargetToHost => {
                    unix_to_windows(&mut path);
                }
            }
            Cow::Owned(OsString::from_wide(&path))
        };
        #[cfg(unix)]
        return if target_os == "windows" {
            // Windows target, Unix host.
            let mut path: Vec<u8> = os_str.into_owned().into_encoded_bytes();
            match direction {
                PathConversion::HostToTarget => {
                    unix_to_windows(&mut path);
                }
                PathConversion::TargetToHost => {
                    windows_to_unix(&mut path);
                }
            }
            Cow::Owned(OsString::from_vec(path))
        } else {
            // Unix-on-Unix, all is fine.
            os_str
        };
    }
}
