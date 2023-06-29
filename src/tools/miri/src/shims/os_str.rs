use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(windows)]
use std::os::windows::ffi::{OsStrExt, OsStringExt};

use rustc_middle::ty::layout::LayoutOf;

use crate::*;

/// Represent how path separator conversion should be done.
pub enum PathConversion {
    HostToTarget,
    TargetToHost,
}

#[cfg(unix)]
pub fn bytes_to_os_str<'tcx>(bytes: &[u8]) -> InterpResult<'tcx, &OsStr> {
    Ok(OsStr::from_bytes(bytes))
}
#[cfg(not(unix))]
pub fn bytes_to_os_str<'tcx>(bytes: &[u8]) -> InterpResult<'tcx, &OsStr> {
    // We cannot use `from_os_str_bytes_unchecked` here since we can't trust `bytes`.
    let s = std::str::from_utf8(bytes)
        .map_err(|_| err_unsup_format!("{:?} is not a valid utf-8 string", bytes))?;
    Ok(OsStr::new(s))
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Helper function to read an OsString from a null-terminated sequence of bytes, which is what
    /// the Unix APIs usually handle.
    fn read_os_str_from_c_str<'a>(
        &'a self,
        ptr: Pointer<Option<Provenance>>,
    ) -> InterpResult<'tcx, &'a OsStr>
    where
        'tcx: 'a,
        'mir: 'a,
    {
        let this = self.eval_context_ref();
        let bytes = this.read_c_str(ptr)?;
        bytes_to_os_str(bytes)
    }

    /// Helper function to read an OsString from a 0x0000-terminated sequence of u16,
    /// which is what the Windows APIs usually handle.
    fn read_os_str_from_wide_str<'a>(
        &'a self,
        ptr: Pointer<Option<Provenance>>,
    ) -> InterpResult<'tcx, OsString>
    where
        'tcx: 'a,
        'mir: 'a,
    {
        #[cfg(windows)]
        pub fn u16vec_to_osstring<'tcx>(u16_vec: Vec<u16>) -> InterpResult<'tcx, OsString> {
            Ok(OsString::from_wide(&u16_vec[..]))
        }
        #[cfg(not(windows))]
        pub fn u16vec_to_osstring<'tcx>(u16_vec: Vec<u16>) -> InterpResult<'tcx, OsString> {
            let s = String::from_utf16(&u16_vec[..])
                .map_err(|_| err_unsup_format!("{:?} is not a valid utf-16 string", u16_vec))?;
            Ok(s.into())
        }

        let u16_vec = self.eval_context_ref().read_wide_str(ptr)?;
        u16vec_to_osstring(u16_vec)
    }

    /// Helper function to write an OsStr as a null-terminated sequence of bytes, which is what
    /// the Unix APIs usually handle. This function returns `Ok((false, length))` without trying
    /// to write if `size` is not large enough to fit the contents of `os_string` plus a null
    /// terminator. It returns `Ok((true, length))` if the writing process was successful. The
    /// string length returned does include the null terminator.
    fn write_os_str_to_c_str(
        &mut self,
        os_str: &OsStr,
        ptr: Pointer<Option<Provenance>>,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        let bytes = os_str.as_os_str_bytes();
        self.eval_context_mut().write_c_str(bytes, ptr, size)
    }

    /// Helper function to write an OsStr as a 0x0000-terminated u16-sequence, which is what the
    /// Windows APIs usually handle.
    ///
    /// If `truncate == false` (the usual mode of operation), this function returns `Ok((false,
    /// length))` without trying to write if `size` is not large enough to fit the contents of
    /// `os_string` plus a null terminator. It returns `Ok((true, length))` if the writing process
    /// was successful. The string length returned does include the null terminator. Length is
    /// measured in units of `u16.`
    ///
    /// If `truncate == true`, then in case `size` is not large enough it *will* write the first
    /// `size.saturating_sub(1)` many items, followed by a null terminator (if `size > 0`).
    fn write_os_str_to_wide_str(
        &mut self,
        os_str: &OsStr,
        ptr: Pointer<Option<Provenance>>,
        size: u64,
        truncate: bool,
    ) -> InterpResult<'tcx, (bool, u64)> {
        #[cfg(windows)]
        fn os_str_to_u16vec<'tcx>(os_str: &OsStr) -> InterpResult<'tcx, Vec<u16>> {
            Ok(os_str.encode_wide().collect())
        }
        #[cfg(not(windows))]
        fn os_str_to_u16vec<'tcx>(os_str: &OsStr) -> InterpResult<'tcx, Vec<u16>> {
            // On non-Windows platforms the best we can do to transform Vec<u16> from/to OS strings is to do the
            // intermediate transformation into strings. Which invalidates non-utf8 paths that are actually
            // valid.
            os_str
                .to_str()
                .map(|s| s.encode_utf16().collect())
                .ok_or_else(|| err_unsup_format!("{:?} is not a valid utf-8 string", os_str).into())
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
        Ok((written, size_needed))
    }

    /// Allocate enough memory to store the given `OsStr` as a null-terminated sequence of bytes.
    fn alloc_os_str_as_c_str(
        &mut self,
        os_str: &OsStr,
        memkind: MemoryKind<MiriMemoryKind>,
    ) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
        let size = u64::try_from(os_str.len()).unwrap().checked_add(1).unwrap(); // Make space for `0` terminator.
        let this = self.eval_context_mut();

        let arg_type = this.tcx.mk_array(this.tcx.types.u8, size);
        let arg_place = this.allocate(this.layout_of(arg_type).unwrap(), memkind)?;
        let (written, _) = self.write_os_str_to_c_str(os_str, arg_place.ptr, size).unwrap();
        assert!(written);
        Ok(arg_place.ptr)
    }

    /// Allocate enough memory to store the given `OsStr` as a null-terminated sequence of `u16`.
    fn alloc_os_str_as_wide_str(
        &mut self,
        os_str: &OsStr,
        memkind: MemoryKind<MiriMemoryKind>,
    ) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
        let size = u64::try_from(os_str.len()).unwrap().checked_add(1).unwrap(); // Make space for `0x0000` terminator.
        let this = self.eval_context_mut();

        let arg_type = this.tcx.mk_array(this.tcx.types.u16, size);
        let arg_place = this.allocate(this.layout_of(arg_type).unwrap(), memkind)?;
        let (written, _) =
            self.write_os_str_to_wide_str(os_str, arg_place.ptr, size, /*truncate*/ false).unwrap();
        assert!(written);
        Ok(arg_place.ptr)
    }

    /// Read a null-terminated sequence of bytes, and perform path separator conversion if needed.
    fn read_path_from_c_str<'a>(
        &'a self,
        ptr: Pointer<Option<Provenance>>,
    ) -> InterpResult<'tcx, Cow<'a, Path>>
    where
        'tcx: 'a,
        'mir: 'a,
    {
        let this = self.eval_context_ref();
        let os_str = this.read_os_str_from_c_str(ptr)?;

        Ok(match this.convert_path(Cow::Borrowed(os_str), PathConversion::TargetToHost) {
            Cow::Borrowed(x) => Cow::Borrowed(Path::new(x)),
            Cow::Owned(y) => Cow::Owned(PathBuf::from(y)),
        })
    }

    /// Read a null-terminated sequence of `u16`s, and perform path separator conversion if needed.
    fn read_path_from_wide_str(
        &self,
        ptr: Pointer<Option<Provenance>>,
    ) -> InterpResult<'tcx, PathBuf> {
        let this = self.eval_context_ref();
        let os_str = this.read_os_str_from_wide_str(ptr)?;

        Ok(this.convert_path(Cow::Owned(os_str), PathConversion::TargetToHost).into_owned().into())
    }

    /// Write a Path to the machine memory (as a null-terminated sequence of bytes),
    /// adjusting path separators if needed.
    fn write_path_to_c_str(
        &mut self,
        path: &Path,
        ptr: Pointer<Option<Provenance>>,
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
        ptr: Pointer<Option<Provenance>>,
        size: u64,
        truncate: bool,
    ) -> InterpResult<'tcx, (bool, u64)> {
        let this = self.eval_context_mut();
        let os_str =
            this.convert_path(Cow::Borrowed(path.as_os_str()), PathConversion::HostToTarget);
        this.write_os_str_to_wide_str(&os_str, ptr, size, truncate)
    }

    /// Allocate enough memory to store a Path as a null-terminated sequence of bytes,
    /// adjusting path separators if needed.
    fn alloc_path_as_c_str(
        &mut self,
        path: &Path,
        memkind: MemoryKind<MiriMemoryKind>,
    ) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
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
        memkind: MemoryKind<MiriMemoryKind>,
    ) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
        let this = self.eval_context_mut();
        let os_str =
            this.convert_path(Cow::Borrowed(path.as_os_str()), PathConversion::HostToTarget);
        this.alloc_os_str_as_wide_str(&os_str, memkind)
    }

    #[allow(clippy::get_first)]
    fn convert_path<'a>(
        &self,
        os_str: Cow<'a, OsStr>,
        direction: PathConversion,
    ) -> Cow<'a, OsStr> {
        let this = self.eval_context_ref();
        let target_os = &this.tcx.sess.target.os;

        #[cfg(windows)]
        return if target_os == "windows" {
            // Windows-on-Windows, all fine.
            os_str
        } else {
            // Unix target, Windows host.
            let (from, to) = match direction {
                PathConversion::HostToTarget => ('\\', '/'),
                PathConversion::TargetToHost => ('/', '\\'),
            };
            let mut converted = os_str
                .encode_wide()
                .map(|wchar| if wchar == from as u16 { to as u16 } else { wchar })
                .collect::<Vec<_>>();
            // We also have to ensure that absolute paths remain absolute.
            match direction {
                PathConversion::HostToTarget => {
                    // If this is an absolute Windows path that starts with a drive letter (`C:/...`
                    // after separator conversion), it would not be considered absolute by Unix
                    // target code.
                    if converted.get(1).copied() == Some(b':' as u16)
                        && converted.get(2).copied() == Some(b'/' as u16)
                    {
                        // We add a `/` at the beginning, to store the absolute Windows
                        // path in something that looks like an absolute Unix path.
                        converted.insert(0, b'/' as u16);
                    }
                }
                PathConversion::TargetToHost => {
                    // If the path is `\C:\`, the leading backslash was probably added by the above code
                    // and we should get rid of it again.
                    if converted.get(0).copied() == Some(b'\\' as u16)
                        && converted.get(2).copied() == Some(b':' as u16)
                        && converted.get(3).copied() == Some(b'\\' as u16)
                    {
                        converted.remove(0);
                    }
                }
            }
            Cow::Owned(OsString::from_wide(&converted))
        };
        #[cfg(unix)]
        return if target_os == "windows" {
            // Windows target, Unix host.
            let (from, to) = match direction {
                PathConversion::HostToTarget => (b'/', b'\\'),
                PathConversion::TargetToHost => (b'\\', b'/'),
            };
            let mut converted = os_str
                .as_bytes()
                .iter()
                .map(|&wchar| if wchar == from { to } else { wchar })
                .collect::<Vec<_>>();
            // We also have to ensure that absolute paths remain absolute.
            match direction {
                PathConversion::HostToTarget => {
                    // If this start withs a `\`, we add `\\?` so it starts with `\\?\` which is
                    // some magic path on Windows that *is* considered absolute.
                    if converted.get(0).copied() == Some(b'\\') {
                        converted.splice(0..0, b"\\\\?".iter().copied());
                    }
                }
                PathConversion::TargetToHost => {
                    // If this starts with `//?/`, it was probably produced by the above code and we
                    // remove the `//?` that got added to get the Unix path back out.
                    if converted.get(0).copied() == Some(b'/')
                        && converted.get(1).copied() == Some(b'/')
                        && converted.get(2).copied() == Some(b'?')
                        && converted.get(3).copied() == Some(b'/')
                    {
                        // Remove first 3 characters
                        converted.splice(0..3, std::iter::empty());
                    }
                }
            }
            Cow::Owned(OsString::from_vec(converted))
        } else {
            // Unix-on-Unix, all is fine.
            os_str
        };
    }
}
