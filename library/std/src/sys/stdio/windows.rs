#![unstable(issue = "none", feature = "windows_stdio")]

use core::str::utf8_char_width;

#[cfg(not(target_vendor = "win7"))]
use crate::alloc::System;
use crate::mem::MaybeUninit;
use crate::os::windows::io::{FromRawHandle, IntoRawHandle};
use crate::sys::handle::Handle;
use crate::sys::pal::api::{self, WinError};
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

pub struct Stdout {
    incomplete_utf8: IncompleteUtf8,
}

pub struct Stderr {
    incomplete_utf8: IncompleteUtf8,
}

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

// The way the console used to work up to Windows 8 is that the "LPC"s carried
// a small scratch space for reads/writes. If the client exceeded it, the server
// allocated a larger buffer from a fixed size section object. Its size was
// `CONSOLE_PORT_MEMORY_SIZE`, aka 64KiB. This meant any IO larger than that failed.
// If the Windows 7 source code ever leaks: The relevant function is `ConsoleAllocateCaptureBuffer`.
//
// It remains a good size for a stack allocated buffer, however.
const MAX_BUFFER_SIZE: usize = 8192;

// The standard buffer size of BufReader for Stdin should be able to hold 3x more bytes than there
// are `u16`'s in MAX_BUFFER_SIZE. This ensures the read data can always be completely decoded from
// UTF-16 to UTF-8.
pub const STDIN_BUF_SIZE: usize = MAX_BUFFER_SIZE / 2 * 3;

#[cfg(not(target_vendor = "win7"))]
const MAX_WRITE_HEAP_LEN: usize = 64 * 1024; // In u16 -> 128KiB

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

/// Returns true if `handle` refers to a proper console (aka conhost).
fn is_console(handle: c::HANDLE) -> bool {
    cfg_select! {
        target_vendor = "win7" => {
            // `GetConsoleMode` will return false (0) if this is a pipe (we don't care about the reported
            // mode). This will only detect Windows Console, not other terminals connected to a pipe like
            // MSYS. Which is exactly what we need, as only Windows Console needs a conversion to UTF-16.
            let mut mode = 0;
            unsafe { c::GetConsoleMode(handle, &mut mode) != 0 }
        }
        _ => unsafe {
            // `GetConsoleMode` incurs a remote RPC round-trip (= bad), so it's best avoided.
            // `NtQueryVolumeInformationFile` lets us test the handle directly.
            // The required `FILE_DEVICE_CONSOLE` type was introduced in Windows 8.
            // Fun fact: `GetFileType` uses `NtQueryVolumeInformationFile` internally.
            let mut io_status = MaybeUninit::<c::IO_STATUS_BLOCK>::uninit();
            let mut device_info = MaybeUninit::<c::FILE_FS_DEVICE_INFORMATION>::uninit();
            let status = c::NtQueryVolumeInformationFile(
                handle,
                io_status.as_mut_ptr(),
                device_info.as_mut_ptr().cast(),
                size_of_val(&device_info) as u32,
                c::FileFsDeviceInformation,
            );
            status >= 0 && device_info.assume_init_ref().DeviceType == c::FILE_DEVICE_CONSOLE
        },
    }
}

fn write(handle_id: u32, data: &[u8], incomplete_utf8: &mut IncompleteUtf8) -> io::Result<usize> {
    if data.is_empty() {
        return Ok(0);
    }

    let handle = get_handle(handle_id)?;

    // At the time of writing, console IO on Windows is high latency:
    // Windows has no TTY buffers, making every call a fully synchronous RPC.
    // So, while always deferring to `write_console_utf16` wastes valuable heap for
    // our UTF16 conversion, it handily beats doing 2 (or more) costly RPC round-trips.
    if !is_console(handle) {
        unsafe {
            let handle = Handle::from_raw_handle(handle);
            let ret = handle.write(data);
            let _ = handle.into_raw_handle(); // Don't close the handle
            ret
        }
    } else {
        write_console_utf16(data, incomplete_utf8, handle)
    }
}

fn write_console_utf16(
    data: &[u8],
    incomplete_utf8: &mut IncompleteUtf8,
    handle: c::HANDLE,
) -> io::Result<usize> {
    // As stated for `MAX_BUFFER_SIZE`, Windows 7 does not support large writes at all.
    // --> Fall back to the small-writes stack-alloc path.
    #[cfg(not(target_vendor = "win7"))]
    if data.len() > MAX_BUFFER_SIZE
        && let capacity = data.len().min(MAX_WRITE_HEAP_LEN)
        && let Ok(mut utf16) = Box::<[u16], _>::try_new_uninit_slice_in(capacity, System)
    {
        return write_console_with_buffer(data, incomplete_utf8, &mut utf16, handle);
    }

    write_console_stack(data, incomplete_utf8, handle)
}

#[cfg_attr(not(target_vendor = "win7"), inline(never))] // Avoid stack allocations unless called
fn write_console_stack(
    data: &[u8],
    incomplete_utf8: &mut IncompleteUtf8,
    handle: c::HANDLE,
) -> io::Result<usize> {
    let mut utf16 = [const { MaybeUninit::uninit() }; MAX_BUFFER_SIZE];
    write_console_with_buffer(data, incomplete_utf8, &mut utf16, handle)
}

fn write_console_with_buffer(
    data: &[u8],
    incomplete_utf8: &mut IncompleteUtf8,
    utf16: &mut [MaybeUninit<u16>],
    handle: c::HANDLE,
) -> io::Result<usize> {
    let mut written = 0;
    while written < data.len() {
        match write_console_chunk(&data[written..], incomplete_utf8, utf16, handle) {
            Ok(0) => break,
            Ok(consumed) => written += consumed,
            Err(error) if written == 0 => return Err(error),
            Err(_) => break,
        }
    }
    Ok(written)
}

fn write_console_chunk(
    data: &[u8],
    incomplete_utf8: &mut IncompleteUtf8,
    utf16: &mut [MaybeUninit<u16>],
    handle: c::HANDLE,
) -> io::Result<usize> {
    debug_assert!(utf16.len() >= 2, "not enough space for a full codepoint");
    debug_assert!(utf16.len() >= 1024, "unreasonably small buffer");

    let mut chunk_len = data.len().min(utf16.len() - incomplete_utf8.len as usize);
    if chunk_len < data.len() {
        chunk_len = incomplete_utf8_suffix_start(&data[..chunk_len]);
    }
    let data = &data[..chunk_len];
    let mut utf16_len = 0;
    let mut remaining = data;

    // Join the incomplete sequence with the new data.
    if incomplete_utf8.len > 0 {
        let mut len = incomplete_utf8.len as usize;
        let width = utf8_char_width(incomplete_utf8.bytes[0]);

        // Accumulate until we have a complete sequence,
        // or something recognizable as a invalid sequence.
        while len < width
            && let Some((&b, rest)) = remaining.split_first()
            && b & 0xc0 == 0x80
        {
            incomplete_utf8.bytes[len] = b;
            len += 1;
            remaining = rest;
        }

        // If we consumed all of `data` without completing the sequence we're done.
        if len < width && remaining.is_empty() {
            incomplete_utf8.len = len as u8;
            return Ok(data.len());
        }

        utf16_len = utf8_to_utf16(&incomplete_utf8.bytes[..len], utf16)?;
    }

    // Strip off any incomplete, trailing UTF8 sequence.
    let (remaining, trailing) = remaining.split_at(incomplete_utf8_suffix_start(remaining));

    // Convert the bulk of the data to UTF16.
    if !remaining.is_empty() {
        utf16_len += utf8_to_utf16(remaining, &mut utf16[utf16_len..])?;
    }

    if utf16_len != 0 {
        // NOTE: WriteConsoleW does not result in short writes.
        let mut written = 0;
        cvt(unsafe {
            c::WriteConsoleW(
                handle,
                utf16.as_ptr().cast(),
                utf16_len as u32,
                &mut written,
                ptr::null_mut(),
            )
        })?;
    }

    incomplete_utf8.bytes[..trailing.len()].copy_from_slice(trailing);
    incomplete_utf8.len = trailing.len() as u8;
    Ok(data.len())
}

fn incomplete_utf8_suffix_start(bytes: &[u8]) -> usize {
    for length in 1..=3.min(bytes.len()) {
        let start = bytes.len() - length;
        let byte = bytes[start];
        if byte & 0xc0 != 0x80 {
            return if utf8_char_width(byte) > length { start } else { bytes.len() };
        }
    }
    bytes.len()
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
            let mut utf16_buf = [MaybeUninit::new(0); 2];
            // Read one u16 character.
            let read = read_u16s_fixup_surrogates(handle, &mut utf16_buf, 1, &mut self.surrogate)?;
            // Read bytes, using the (now-empty) self.incomplete_utf8 as extra space.
            let read_bytes = utf16_to_utf8(
                unsafe { utf16_buf[..read].assume_init_ref() },
                &mut self.incomplete_utf8.bytes,
            )?;

            // Read in the bytes from incomplete_utf8 until the buffer is full.
            self.incomplete_utf8.len = read_bytes as u8;
            // No-op if no bytes.
            bytes_copied += self.incomplete_utf8.read(&mut buf[bytes_copied..]);
            Ok(bytes_copied)
        } else {
            // Decode into the space after the bytes copied from `incomplete_utf8` (if any), so
            // they are not overwritten; this also makes `amount` below account for the space
            // that is actually still available.
            let buf = &mut buf[bytes_copied..];
            let mut utf16_buf = [MaybeUninit::<u16>::uninit(); MAX_BUFFER_SIZE / 2];

            // In the worst case, a UTF-8 string can take 3 bytes for every `u16` of a UTF-16. So
            // we can read at most a third of `buf.len()` chars and uphold the guarantee no data gets
            // lost.
            let amount = cmp::min(buf.len() / 3, utf16_buf.len());
            let read =
                read_u16s_fixup_surrogates(handle, &mut utf16_buf, amount, &mut self.surrogate)?;
            // Safety `read_u16s_fixup_surrogates` returns the number of items
            // initialized.
            let utf16s = unsafe { utf16_buf[..read].assume_init_ref() };
            match utf16_to_utf8(utf16s, buf) {
                Ok(value) => Ok(bytes_copied + value),
                Err(e) => Err(e),
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
        nLength: size_of::<c::CONSOLE_READCONSOLE_CONTROL>() as u32,
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
            c::CP_UTF8,         // CodePage
            0,                  // dwFlags
            utf16.as_ptr(),     // lpWideCharStr
            utf16.len() as i32, // cchWideChar
            utf8.as_mut_ptr(),  // lpMultiByteStr
            utf8.len() as i32,  // cbMultiByte
            ptr::null(),        // lpDefaultChar
            ptr::null_mut(),    // lpUsedDefaultChar
        )
    };
    if result == 0 {
        return Err(io::Error::last_os_error());
    } else {
        Ok(result as usize)
    }
}

fn utf8_to_utf16(data: &[u8], utf16: &mut [MaybeUninit<u16>]) -> io::Result<usize> {
    debug_assert!(!data.is_empty());

    if utf16.is_empty() {
        return Ok(0);
    }

    // Fun fact: As is often the case, Windows' algorithm predates Unicode's recommendation.
    // The current behavior was implemented for Windows Vista (2007) and Unicode's recommendation
    // for "Best Practices for Using U+FFFD" came with Unicode 5.2 (2009) [^1]. So, now if a
    // continuation byte violates a boundary limit, it should restore it [^2], but this consumes it.
    // For instance, `ED A0 80` produces two U+FFFD with this API, while it should be three.
    // In practice this difference bears no meaning for console output.
    //
    // NOTE: However, it means we have to be careful about the treatment of `incomplete_utf8`,
    // because the behavior of `MultiByteToWideChar` could theoretically change in the future.
    //
    // `MultiByteToWideChar` is technically a fancy wrapper around `RtlUTF8ToUnicodeN`,
    // whose MSDN is a bit more enlightening about the behavior of the conversion.
    // Despite its name, it's not a syscall, but a simple function inside ntdll.dll.
    //
    // [^1]: https://www.unicode.org/review/pr-121.html
    // [^2]: https://encoding.spec.whatwg.org/#utf-8-decoder
    let len = unsafe {
        c::MultiByteToWideChar(
            c::CP_UTF8,
            0,
            data.as_ptr().cast(),
            data.len() as i32,
            utf16.as_mut_ptr().cast(),
            utf16.len() as i32,
        )
    };
    if len <= 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(len as usize)
}

impl IncompleteUtf8 {
    pub const fn new() -> IncompleteUtf8 {
        IncompleteUtf8 { bytes: [0; char::MAX_LEN_UTF8], len: 0 }
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout { incomplete_utf8: IncompleteUtf8::new() }
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write(c::STD_OUTPUT_HANDLE, buf, &mut self.incomplete_utf8)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr { incomplete_utf8: IncompleteUtf8::new() }
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write(c::STD_ERROR_HANDLE, buf, &mut self.incomplete_utf8)
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
