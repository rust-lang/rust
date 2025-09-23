use crate::ffi::OsString;
use crate::io;
use crate::os::unix::ffi::OsStringExt;
use crate::sys::pal::os::errno;

pub fn hostname() -> io::Result<OsString> {
    // Query the system for the maximum host name length.
    let host_name_max = match unsafe { libc::sysconf(libc::_SC_HOST_NAME_MAX) } {
        // If this fails (possibly because there is no maximum length), then
        // assume a maximum length of _POSIX_HOST_NAME_MAX (255).
        -1 => 255,
        max => max as usize,
    };

    // Reserve space for the nul terminator too.
    let mut buf = Vec::<u8>::try_with_capacity(host_name_max + 1)?;
    loop {
        // SAFETY: `buf.capacity()` bytes of `buf` are writable.
        let r = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.capacity()) };
        match (r != 0).then(errno) {
            None => {
                // Unfortunately, the UNIX specification says that the name will
                // be truncated if it does not fit in the buffer, without returning
                // an error. As additionally, the truncated name may still be null-
                // terminated, there is no reliable way to  detect truncation.
                // Fortunately, most platforms ignore what the specification says
                // and return an error (mostly ENAMETOOLONG). Should that not be
                // the case, the following detects truncation if the null-terminator
                // was omitted. Note that this check does not impact performance at
                // all as we need to find the length of the string anyways.
                //
                // Use `strnlen` as it does not place an initialization requirement
                // on the bytes after the nul terminator.
                //
                // SAFETY: `buf.capacity()` bytes of `buf` are accessible, and are
                // initialized up to and including a possible nul terminator.
                let len = unsafe { libc::strnlen(buf.as_ptr().cast(), buf.capacity()) };
                if len < buf.capacity() {
                    // If the string is nul-terminated, we assume that is has not
                    // been truncated, as the capacity *should be* enough to hold
                    // `HOST_NAME_MAX` bytes.
                    // SAFETY: `len + 1` bytes have been initialized (we exclude
                    // the nul terminator from the string).
                    unsafe { buf.set_len(len) };
                    return Ok(OsString::from_vec(buf));
                }
            }
            // As `buf.capacity()` is always less than or equal to `isize::MAX`
            // (Rust allocations cannot exceed that limit), the only way `EINVAL`
            // can be returned is if the system uses `EINVAL` to report that the
            // name does not fit in the provided buffer. In that case (or in the
            // case of `ENAMETOOLONG`), resize the buffer and try again.
            Some(libc::EINVAL | libc::ENAMETOOLONG) => {}
            // Other error codes (e.g. EPERM) have nothing to do with the buffer
            // size and should be returned to the user.
            Some(err) => return Err(io::Error::from_raw_os_error(err)),
        }

        // Resize the buffer (according to `Vec`'s resizing rules) and try again.
        buf.try_reserve(buf.capacity() + 1)?;
    }
}
