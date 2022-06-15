use crate::{io, os::uefi};
use r_efi::efi;
use r_efi::protocols::{simple_text_input, simple_text_output};
use r_efi::signatures::system::boot_services::WaitForEventSignature;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

const MAX_BUFFER_SIZE: usize = 8192;

pub const STDIN_BUF_SIZE: usize = MAX_BUFFER_SIZE / 2 * 3;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin(())
    }

    unsafe fn fire_wait_event(
        con_in: *mut simple_text_input::Protocol,
        wait_for_event: WaitForEventSignature,
    ) -> io::Result<()> {
        let r = unsafe {
            let mut x: usize = 0;
            (wait_for_event)(1, &mut (*con_in).wait_for_key, &mut x)
        };

        if r.is_error() {
            Err(io::Error::new(io::ErrorKind::Other, "Could not wait for event"))
        } else {
            Ok(())
        }
    }

    unsafe fn read_key_stroke(con_in: *mut simple_text_input::Protocol) -> io::Result<u16> {
        let mut input_key = simple_text_input::InputKey::default();
        let r = unsafe { ((*con_in).read_key_stroke)(con_in, &mut input_key) };

        if r.is_error() || input_key.scan_code != 0 {
            Err(io::Error::new(io::ErrorKind::InvalidInput, "Error in Reading Keystroke"))
        } else {
            Ok(input_key.unicode_char)
        }
    }

    unsafe fn reset_weak(con_in: *mut simple_text_input::Protocol) -> io::Result<()> {
        let r = unsafe { ((*con_in).reset)(con_in, efi::Boolean::TRUE) };

        if r.is_error() {
            Err(io::Error::new(io::ErrorKind::InvalidInput, "Device Error"))
        } else {
            Ok(())
        }
    }

    unsafe fn write_character(
        con_out: *mut simple_text_output::Protocol,
        character: u16,
    ) -> io::Result<()> {
        let mut buf: [u16; 2] = [character, 0];
        let r = unsafe { ((*con_out).output_string)(con_out, buf.as_mut_ptr()) };
        if r.is_error() {
            Err(io::Error::new(io::ErrorKind::InvalidInput, "Device Error"))
        } else if character == u16::from(b'\r') {
            // Handle enter
            unsafe { Self::write_character(con_out, u16::from(b'\n')) }
        } else {
            Ok(())
        }
    }
}

impl io::Read for Stdin {
    // Reads 1 UCS-2 character at a time and returns.
    // FIXME: Implement buffered reading. Currently backspace and other characters are read as
    // normal characters. Thus it might look like line-editing but it actually isn't
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let con_in = unsafe { get_con_in() }?;
        let con_out = unsafe { get_con_out() }?;
        let wait_for_event = unsafe { get_wait_for_event() }?;

        if buf.len() < 3 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Buffer too small"));
        }

        let bytes_read = {
            unsafe { Stdin::reset_weak(con_in) }?;
            unsafe { Stdin::fire_wait_event(con_in, wait_for_event) }?;
            let ch = unsafe { Stdin::read_key_stroke(con_in) }?;
            unsafe { Stdin::write_character(con_out, ch) }?;

            utf16_to_utf8_char(ch, buf)
        };

        Ok(bytes_read)
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout(())
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let con_out = unsafe { get_con_out() }?;
        simple_text_output_write(con_out, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr(())
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let std_err = unsafe { get_std_err() }?;
        simple_text_output_write(std_err, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(efi::Status::DEVICE_ERROR.as_usize() as i32)
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}

fn utf8_to_utf16(utf8_buf: &[u8], utf16_buf: &mut [u16]) -> io::Result<usize> {
    let utf8_buf_len = utf8_buf.len();
    let utf16_buf_len = utf16_buf.len();
    let mut utf8_buf_count = 0;
    let mut utf16_buf_count = 0;

    // Since it is possible for the bytes written to be <= the utf8_buf, we just stop writing if
    // the utf16_buf fills up
    // Also leave space for null termination in utf16_buf
    while utf8_buf_count < utf8_buf_len && utf16_buf_count + 1 < utf16_buf_len {
        match utf8_buf[utf8_buf_count] {
            0b0000_0000..0b0111_1111 => {
                // 1-byte

                // Convert LF to CRLF
                if utf8_buf[utf8_buf_count] == b'\n' {
                    utf16_buf[utf16_buf_count] = u16::from(b'\r');
                    utf16_buf_count += 1;
                }

                utf16_buf[utf16_buf_count] = u16::from(utf8_buf[utf8_buf_count]);

                utf8_buf_count += 1;
            }
            0b1100_0000..0b1101_1111 => {
                // 2-byte
                if utf16_buf_count + 1 >= utf8_buf_len {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8"));
                }
                let a = u16::from(utf8_buf[utf8_buf_count] & 0b0001_1111);
                let b = u16::from(utf8_buf[utf8_buf_count + 1] & 0b0011_1111);
                utf16_buf[utf16_buf_count] = a << 6 | b;

                utf8_buf_count += 2;
            }
            0b1110_0000..0b1110_1111 => {
                // 3-byte
                if utf16_buf_count + 2 >= utf8_buf_len {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8"));
                }
                let a = u16::from(utf8_buf[utf8_buf_count] & 0b0000_1111);
                let b = u16::from(utf8_buf[utf8_buf_count + 1] & 0b0011_1111);
                let c = u16::from(utf8_buf[utf8_buf_count + 2] & 0b0011_1111);
                utf16_buf[utf16_buf_count] = a << 12 | b << 6 | c;
                utf8_buf_count += 3;
            }
            0b1111_0000..0b1111_0111 => {
                // 4-byte
                // We just print a restricted Character
                utf16_buf[utf16_buf_count] = 0xfffdu16;
                utf8_buf_count += 4;
            }
            _ => {
                // Invalid Data
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8"));
            }
        }

        utf16_buf_count += 1;
    }

    utf16_buf[utf16_buf_count] = 0;

    Ok(utf8_buf_count)
}

fn utf16_to_utf8_char(ch: u16, buf: &mut [u8]) -> usize {
    match ch {
        0b0000_0000_0000_0000..0b0000_0000_0111_1111 => {
            // 1-byte

            // Convert CR to LF
            buf[0] = if ch == u16::from(b'\r') { b'\n' } else { ch as u8 };
            1
        }
        0b0000_0000_0111_1111..0b0000_0111_1111_1111 => {
            // 2-byte
            let a = ((ch & 0b0000_0111_1100_0000) >> 6) as u8;
            let b = (ch & 0b0000_0000_0011_1111) as u8;
            buf[0] = a | 0b1100_0000;
            buf[1] = b | 0b1000_0000;
            2
        }
        _ => {
            // 3-byte
            let a = ((ch & 0b1111_0000_0000_0000) >> 12) as u8;
            let b = ((ch & 0b0000_1111_1100_0000) >> 6) as u8;
            let c = (ch & 0b0000_0000_0011_1111) as u8;
            buf[0] = a | 0b1110_0000;
            buf[1] = b | 0b1000_0000;
            buf[2] = c | 0b1000_0000;
            3
        }
    }
}

fn simple_text_output_write(
    protocol: *mut simple_text_output::Protocol,
    buf: &[u8],
) -> io::Result<usize> {
    let output_string_ptr = unsafe { (*protocol).output_string };

    let mut output = [0u16; MAX_BUFFER_SIZE / 2];
    let count = utf8_to_utf16(buf, &mut output)?;

    let r = (output_string_ptr)(protocol, output.as_mut_ptr());

    if r.is_error() {
        Err(io::Error::new(io::ErrorKind::Other, r.as_usize().to_string()))
    } else {
        Ok(count)
    }
}

// Returns error if `SystemTable->ConIn` is null.
unsafe fn get_con_in() -> io::Result<*mut simple_text_input::Protocol> {
    let st = unsafe {
        match uefi::env::get_system_table() {
            Ok(x) => x,
            Err(_) => {
                return Err(io::Error::new(io::ErrorKind::NotFound, "Global System Table"));
            }
        }
    };

    let con_in = unsafe { (*st).con_in };

    if con_in.is_null() {
        Err(io::Error::new(io::ErrorKind::NotFound, "ConIn"))
    } else {
        Ok(con_in)
    }
}

unsafe fn get_wait_for_event() -> io::Result<WaitForEventSignature> {
    let st = unsafe {
        match uefi::env::get_system_table() {
            Ok(x) => x,
            Err(_) => {
                return Err(io::Error::new(io::ErrorKind::NotFound, "Global System Table"));
            }
        }
    };

    let boot_services = unsafe { (*st).boot_services };

    if boot_services.is_null() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "Boot Services"));
    }

    Ok(unsafe { (*boot_services).wait_for_event })
}

// Returns error if `SystemTable->ConOut` is null.
unsafe fn get_con_out() -> io::Result<*mut simple_text_output::Protocol> {
    let st = unsafe {
        match uefi::env::get_system_table() {
            Ok(x) => x,
            Err(_) => {
                return Err(io::Error::new(io::ErrorKind::NotFound, "Global System Table"));
            }
        }
    };

    let con_out = unsafe { (*st).con_out };

    if con_out.is_null() {
        Err(io::Error::new(io::ErrorKind::NotFound, "ConOut"))
    } else {
        Ok(con_out)
    }
}

// Returns error if `SystemTable->StdErr` is null.
unsafe fn get_std_err() -> io::Result<*mut simple_text_output::Protocol> {
    let st = unsafe {
        match uefi::env::get_system_table() {
            Ok(x) => x,
            Err(_) => {
                return Err(io::Error::new(io::ErrorKind::NotFound, "Global System Table"));
            }
        }
    };

    let std_err = unsafe { (*st).std_err };

    if std_err.is_null() {
        Err(io::Error::new(io::ErrorKind::NotFound, "StdErr"))
    } else {
        Ok(std_err)
    }
}
