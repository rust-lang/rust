#[cfg(test)]
mod tests;

#[cfg(not(target_os = "espidf"))]
pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// Returns the value for [`confstr(key, ...)`][posix_confstr]. Currently only
/// used on Darwin, but should work on any unix (in case we need to get
/// `_CS_PATH` or `_CS_V[67]_ENV` in the future).
///
/// [posix_confstr]:
///     https://pubs.opengroup.org/onlinepubs/9699919799/functions/confstr.html
//
// FIXME: Support `confstr` in Miri.
#[cfg(all(target_vendor = "apple", not(miri)))]
pub fn confstr(
    key: crate::ffi::c_int,
    size_hint: Option<usize>,
) -> crate::io::Result<crate::ffi::OsString> {
    use crate::ffi::OsString;
    use crate::io;
    use crate::os::unix::ffi::OsStringExt;

    let mut buf: Vec<u8> = Vec::with_capacity(0);
    let mut bytes_needed_including_nul = size_hint
        .unwrap_or_else(|| {
            // Treat "None" as "do an extra call to get the length". In theory
            // we could move this into the loop below, but it's hard to do given
            // that it isn't 100% clear if it's legal to pass 0 for `len` when
            // the buffer isn't null.
            unsafe { libc::confstr(key, core::ptr::null_mut(), 0) }
        })
        .max(1);
    // If the value returned by `confstr` is greater than the len passed into
    // it, then the value was truncated, meaning we need to retry. Note that
    // while `confstr` results don't seem to change for a process, it's unclear
    // if this is guaranteed anywhere, so looping does seem required.
    while bytes_needed_including_nul > buf.capacity() {
        // We write into the spare capacity of `buf`. This lets us avoid
        // changing buf's `len`, which both simplifies `reserve` computation,
        // allows working with `Vec<u8>` instead of `Vec<MaybeUninit<u8>>`, and
        // may avoid a copy, since the Vec knows that none of the bytes are needed
        // when reallocating (well, in theory anyway).
        buf.reserve(bytes_needed_including_nul);
        // `confstr` returns
        // - 0 in the case of errors: we break and return an error.
        // - The number of bytes written, iff the provided buffer is enough to
        //   hold the entire value: we break and return the data in `buf`.
        // - Otherwise, the number of bytes needed (including nul): we go
        //   through the loop again.
        bytes_needed_including_nul =
            unsafe { libc::confstr(key, buf.as_mut_ptr().cast(), buf.capacity()) };
    }
    // `confstr` returns 0 in the case of an error.
    if bytes_needed_including_nul == 0 {
        return Err(io::Error::last_os_error());
    }
    // Safety: `confstr(..., buf.as_mut_ptr(), buf.capacity())` returned a
    // non-zero value, meaning `bytes_needed_including_nul` bytes were
    // initialized.
    unsafe {
        buf.set_len(bytes_needed_including_nul);
        // Remove the NUL-terminator.
        let last_byte = buf.pop();
        // ... and smoke-check that it *was* a NUL-terminator.
        assert_eq!(last_byte, Some(0), "`confstr` provided a string which wasn't nul-terminated");
    };
    Ok(OsString::from_vec(buf))
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn glibc_version() -> Option<(usize, usize)> {
    use crate::ffi::CStr;

    unsafe extern "C" {
        fn gnu_get_libc_version() -> *const libc::c_char;
    }
    let version_cstr = unsafe { CStr::from_ptr(gnu_get_libc_version()) };
    if let Ok(version_str) = version_cstr.to_str() {
        parse_glibc_version(version_str)
    } else {
        None
    }
}

/// Returns Some((major, minor)) if the string is a valid "x.y" version,
/// ignoring any extra dot-separated parts. Otherwise return None.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn parse_glibc_version(version: &str) -> Option<(usize, usize)> {
    let mut parsed_ints = version.split('.').map(str::parse::<usize>).fuse();
    match (parsed_ints.next(), parsed_ints.next()) {
        (Some(Ok(major)), Some(Ok(minor))) => Some((major, minor)),
        _ => None,
    }
}
