#![unstable(issue = "none", feature = "windows_stdio")]

use super::api::{self, WinError};
use crate::mem::MaybeUninit;
use crate::os::windows::io::{FromRawHandle, IntoRawHandle};
use crate::sys::handle::Handle;
use crate::sys::{c, cvt};
use crate::{cmp, io, ptr};

#[cfg(test)]
mod tests;

// Don't cache handles but get them fresh for every read/write. This allows us to track changes to
// the value over time (such as if a process calls `SetStdHandle` while it's running). See #40490.
pub struct Stdin {
    surrogate: u16,
    incomplete_utf8: IncompleteUtf8,
}

pub struct Stdout {}

pub struct Stderr {}

struct IncompleteUtf8 {
    bytes: [u8; 4],
    len: u8,
}

impl IncompleteUtf8 {
    // Implemented for use in Stdin::read.
    fn read(&mut self, buf: &mut [u8]) -> usize {
        // Write to buffer until the buffer is full or we run out of bytes.
        let to_write = cmp::min(buf.len(), self.len as usize);
        buf[..to_write].copy_from_slice(&self.bytes[..to_write]);

        // Rotate the remaining bytes if not enough remaining space in buffer.
        if usize::from(self.len) > buf.len() {
            self.bytes.copy_within(to_write.., 0);
            self.len -= to_write as u8;
        } else {
            self.len = 0;
        }

        to_write
    }
}

// Apparently Windows doesn't handle large reads on stdin or writes to stdout/stderr well (see
// #13304 for details).
//
// From MSDN (2011): "The storage for this buffer is allocated from a shared heap for the
// process that is 64 KB in size. The maximum size of the buffer will depend on heap usage."
//
// We choose the cap at 8 KiB because libuv does the same, and it seems to be acceptable so far.
const MAX_BUFFER_SIZE: usize = 8192;

// The standard buffer size of BufReader for Stdin should be able to hold 3x more bytes than there
// are `u16`'s in MAX_BUFFER_SIZE. This ensures the read data can always be completely decoded from
// UTF-16 to UTF-8.
pub const STDIN_BUF_SIZE: usize = MAX_BUFFER_SIZE / 2 * 3;

pub fn get_handle(handle_id: u32) -> io::Result<c::HANDLE> {
    let handle = unsafe { c::GetStdHandle(handle_id) };
    if handle == c::INVALID_HANDLE_VALUE {
        Err(io::Error::last_os_error())
    } else if handle.is_null() {
        Err(io::Error::from_raw_os_error(c::ERROR_INVALID_HANDLE as i32))
    } else {
        Ok(handle)
    }
}

fn is_console(handle: c::HANDLE) -> bool {
    // `GetConsoleMode` will return false (0) if this is a pipe (we don't care about the reported
    // mode). This will only detect Windows Console, not other terminals connected to a pipe like
    // MSYS. Which is exactly what we need, as only Windows Console needs a conversion to UTF-16.
    let mut mode = 0;
    unsafe { c::GetConsoleMode(handle, &mut mode) != 0 }
}

/// Returns true if the attached console's code page is currently UTF-8.
#[cfg(not(target_vendor = "win7"))]
fn is_utf8_console() -> bool {
    unsafe { c::GetConsoleOutputCP() == c::CP_UTF8 }
}

#[cfg(target_vendor = "win7")]
fn is_utf8_console() -> bool {
    // Windows 7 has a fun "feature" where WriteFile on a console handle will return
    // the number of UTF-16 code units written and not the number of bytes from the input string.
    // So we always claim the console isn't UTF-8 to trigger the WriteConsole fallback code.
    false
}

fn write(handle_id: u32, data: &[u8]) -> io::Result<usize> {
    if data.is_empty() {
        return Ok(0);
    }

    let handle = get_handle(handle_id)?;
    if !is_console(handle) || is_utf8_console() {
        unsafe {
            let handle = Handle::from_raw_handle(handle);
            let ret = handle.write(data);
            let _ = handle.into_raw_handle(); // Don't close the handle
            return ret;
        }
    } else {
        write_console_utf16(data, handle)
    }
}

fn write_console_utf16(data: &[u8], handle: c::HANDLE) -> io::Result<usize> {
    let mut buffer = [MaybeUninit::<u16>::uninit(); MAX_BUFFER_SIZE / 2];
    let data = &data[..data.len().min(buffer.len())];

    // Split off any trailing incomplete UTF-8 from the end of the input.
    let utf8 = trim_last_char_boundary(data);
    let utf16 = utf8_to_utf16_lossy(utf8, &mut buffer);
    debug_assert!(!utf16.is_empty());

    // Write the UTF-16 chars to the console.
    // This will succeed in one write so long as our [u16] slice is smaller than the console's buffer,
    // which we've ensured by truncating the input (see `MAX_BUFFER_SIZE`).
    let written = write_u16s(handle, &utf16)?;
    debug_assert_eq!(written, utf16.len());
    Ok(utf8.len())
}

fn utf8_to_utf16_lossy<'a>(utf8: &[u8], utf16: &'a mut [MaybeUninit<u16>]) -> &'a [u16] {
    unsafe {
        let result = c::MultiByteToWideChar(
            c::CP_UTF8,                          // CodePage
            0,                                   // dwFlags
            utf8.as_ptr(),                       // lpMultiByteStr
            utf8.len() as i32,                   // cbMultiByte
            utf16.as_mut_ptr() as *mut c::WCHAR, // lpWideCharStr
            utf16.len() as i32,                  // cchWideChar
        );
        // The only way an error can happen here is if we've messed up.
        debug_assert!(result != 0, "Unexpected error in MultiByteToWideChar");
        // Safety: MultiByteToWideChar initializes `result` values.
        MaybeUninit::slice_assume_init_ref(&utf16[..result as usize])
    }
}

fn write_u16s(handle: c::HANDLE, data: &[u16]) -> io::Result<usize> {
    debug_assert!(data.len() < u32::MAX as usize);
    let mut written = 0;
    cvt(unsafe {
        c::WriteConsoleW(handle, data.as_ptr(), data.len() as u32, &mut written, ptr::null_mut())
    })?;
    Ok(written as usize)
}

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin { surrogate: 0, incomplete_utf8: IncompleteUtf8::new() }
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let handle = get_handle(c::STD_INPUT_HANDLE)?;
        if !is_console(handle) {
            unsafe {
                let handle = Handle::from_raw_handle(handle);
                let ret = handle.read(buf);
                let _ = handle.into_raw_handle(); // Don't close the handle
                return ret;
            }
        }

        // If there are bytes in the incomplete utf-8, start with those.
        // (No-op if there is nothing in the buffer.)
        let mut bytes_copied = self.incomplete_utf8.read(buf);

        if bytes_copied == buf.len() {
            Ok(bytes_copied)
        } else if buf.len() - bytes_copied < 4 {
            // Not enough space to get a UTF-8 byte. We will use the incomplete UTF8.
            let mut utf16_buf = [MaybeUninit::new(0); 1];
            // Read one u16 character.
            let read = read_u16s_fixup_surrogates(handle, &mut utf16_buf, 1, &mut self.surrogate)?;
            // Read bytes, using the (now-empty) self.incomplete_utf8 as extra space.
            let read_bytes = utf16_to_utf8(
                unsafe { MaybeUninit::slice_assume_init_ref(&utf16_buf[..read]) },
                &mut self.incomplete_utf8.bytes,
            )?;

            // Read in the bytes from incomplete_utf8 until the buffer is full.
            self.incomplete_utf8.len = read_bytes as u8;
            // No-op if no bytes.
            bytes_copied += self.incomplete_utf8.read(&mut buf[bytes_copied..]);
            Ok(bytes_copied)
        } else {
            let mut utf16_buf = [MaybeUninit::<u16>::uninit(); MAX_BUFFER_SIZE / 2];

            // In the worst case, a UTF-8 string can take 3 bytes for every `u16` of a UTF-16. So
            // we can read at most a third of `buf.len()` chars and uphold the guarantee no data gets
            // lost.
            let amount = cmp::min(buf.len() / 3, utf16_buf.len());
            let read =
                read_u16s_fixup_surrogates(handle, &mut utf16_buf, amount, &mut self.surrogate)?;
            // Safety `read_u16s_fixup_surrogates` returns the number of items
            // initialized.
            let utf16s = unsafe { MaybeUninit::slice_assume_init_ref(&utf16_buf[..read]) };
            match utf16_to_utf8(utf16s, buf) {
                Ok(value) => return Ok(bytes_copied + value),
                Err(e) => return Err(e),
            }
        }
    }
}

// We assume that if the last `u16` is an unpaired surrogate they got sliced apart by our
// buffer size, and keep it around for the next read hoping to put them together.
// This is a best effort, and might not work if we are not the only reader on Stdin.
fn read_u16s_fixup_surrogates(
    handle: c::HANDLE,
    buf: &mut [MaybeUninit<u16>],
    mut amount: usize,
    surrogate: &mut u16,
) -> io::Result<usize> {
    // Insert possibly remaining unpaired surrogate from last read.
    let mut start = 0;
    if *surrogate != 0 {
        buf[0] = MaybeUninit::new(*surrogate);
        *surrogate = 0;
        start = 1;
        if amount == 1 {
            // Special case: `Stdin::read` guarantees we can always read at least one new `u16`
            // and combine it with an unpaired surrogate, because the UTF-8 buffer is at least
            // 4 bytes.
            amount = 2;
        }
    }
    let mut amount = read_u16s(handle, &mut buf[start..amount])? + start;

    if amount > 0 {
        // Safety: The returned `amount` is the number of values initialized,
        // and it is not 0, so we know that `buf[amount - 1]` have been
        // initialized.
        let last_char = unsafe { buf[amount - 1].assume_init() };
        if matches!(last_char, 0xD800..=0xDBFF) {
            // high surrogate
            *surrogate = last_char;
            amount -= 1;
        }
    }
    Ok(amount)
}

// Returns `Ok(n)` if it initialized `n` values in `buf`.
fn read_u16s(handle: c::HANDLE, buf: &mut [MaybeUninit<u16>]) -> io::Result<usize> {
    // Configure the `pInputControl` parameter to not only return on `\r\n` but also Ctrl-Z, the
    // traditional DOS method to indicate end of character stream / user input (SUB).
    // See #38274 and https://stackoverflow.com/questions/43836040/win-api-readconsole.
    const CTRL_Z: u16 = 0x1A;
    const CTRL_Z_MASK: u32 = 1 << CTRL_Z;
    let input_control = c::CONSOLE_READCONSOLE_CONTROL {
        nLength: crate::mem::size_of::<c::CONSOLE_READCONSOLE_CONTROL>() as u32,
        nInitialChars: 0,
        dwCtrlWakeupMask: CTRL_Z_MASK,
        dwControlKeyState: 0,
    };

    let mut amount = 0;
    loop {
        cvt(unsafe {
            c::SetLastError(0);
            c::ReadConsoleW(
                handle,
                buf.as_mut_ptr() as *mut core::ffi::c_void,
                buf.len() as u32,
                &mut amount,
                &input_control,
            )
        })?;

        // ReadConsoleW returns success with ERROR_OPERATION_ABORTED for Ctrl-C or Ctrl-Break.
        // Explicitly check for that case here and try again.
        if amount == 0 && api::get_last_error() == WinError::OPERATION_ABORTED {
            continue;
        }
        break;
    }
    // Safety: if `amount > 0`, then that many bytes were written, so
    // `buf[amount as usize - 1]` has been initialized.
    if amount > 0 && unsafe { buf[amount as usize - 1].assume_init() } == CTRL_Z {
        amount -= 1;
    }
    Ok(amount as usize)
}

fn utf16_to_utf8(utf16: &[u16], utf8: &mut [u8]) -> io::Result<usize> {
    debug_assert!(utf16.len() <= i32::MAX as usize);
    debug_assert!(utf8.len() <= i32::MAX as usize);

    if utf16.is_empty() {
        return Ok(0);
    }

    let result = unsafe {
        c::WideCharToMultiByte(
            c::CP_UTF8,              // CodePage
            c::WC_ERR_INVALID_CHARS, // dwFlags
            utf16.as_ptr(),          // lpWideCharStr
            utf16.len() as i32,      // cchWideChar
            utf8.as_mut_ptr(),       // lpMultiByteStr
            utf8.len() as i32,       // cbMultiByte
            ptr::null(),             // lpDefaultChar
            ptr::null_mut(),         // lpUsedDefaultChar
        )
    };
    if result == 0 {
        // We can't really do any better than forget all data and return an error.
        Err(io::const_error!(
            io::ErrorKind::InvalidData,
            "Windows stdin in console mode does not support non-UTF-16 input; \
            encountered unpaired surrogate",
        ))
    } else {
        Ok(result as usize)
    }
}

impl IncompleteUtf8 {
    pub const fn new() -> IncompleteUtf8 {
        IncompleteUtf8 { bytes: [0; 4], len: 0 }
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout {}
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write(c::STD_OUTPUT_HANDLE, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr {}
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write(c::STD_ERROR_HANDLE, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(c::ERROR_INVALID_HANDLE as i32)
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}

/// Trim one incomplete UTF-8 char from the end of a byte slice.
///
/// If trimming would lead to an empty slice then it returns `bytes` instead.
///
/// Note: This function is optimized for size rather than speed.
pub fn trim_last_char_boundary(bytes: &[u8]) -> &[u8] {
    // UTF-8's multiple-byte encoding uses the leading bits to encode the length of a code point.
    // The bits of a multi-byte sequence are (where `n` is a placeholder for any bit):
    //
    // 11110nnn 10nnnnnn 10nnnnnn 10nnnnnn
    // 1110nnnn 10nnnnnn 10nnnnnn
    // 110nnnnn 10nnnnnn
    //
    // So if follows that an incomplete sequence is one of these:
    // 11110nnn 10nnnnnn 10nnnnnn
    // 11110nnn 10nnnnnn
    // 1110nnnn 10nnnnnn
    // 11110nnn
    // 1110nnnn
    // 110nnnnn

    // Get up to three bytes from the end of the slice and encode them as a u32
    // because it turns out the compiler is very good at optimizing numbers.
    let u = match bytes {
        [.., b1, b2, b3] => (*b1 as u32) << 16 | (*b2 as u32) << 8 | *b3 as u32,
        [.., b1, b2] => (*b1 as u32) << 8 | *b2 as u32,
        // If it's just a single byte or empty then we return the full slice
        _ => return bytes,
    };
    if (u & 0b_11111000_11000000_11000000 == 0b_11110000_10000000_10000000) && bytes.len() >= 4 {
        &bytes[..bytes.len() - 3]
    } else if (u & 0b_11111000_11000000 == 0b_11110000_10000000
        || u & 0b_11110000_11000000 == 0b_11100000_10000000)
        && bytes.len() >= 3
    {
        &bytes[..bytes.len() - 2]
    } else if (u & 0b_1111_1000 == 0b_1111_0000
        || u & 0b_11110000 == 0b_11100000
        || u & 0b_11100000 == 0b_11000000)
        && bytes.len() >= 2
    {
        &bytes[..bytes.len() - 1]
    } else {
        bytes
    }
}
