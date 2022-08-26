use super::super::process::uefi_command_protocol;
use super::common::{self, status_to_io_error};
use crate::sys::pipe;
use crate::sys_common::ucs2;
use crate::{io, os::uefi, ptr::NonNull};
use r_efi::protocols::{simple_text_input, simple_text_output};
use r_efi::system::BootWaitForEvent;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

const MAX_BUFFER_SIZE: usize = 8192;

pub const STDIN_BUF_SIZE: usize = MAX_BUFFER_SIZE / 2 * 3;

impl Stdin {
    #[inline]
    pub const fn new() -> Stdin {
        Stdin(())
    }

    // Wait for key input
    unsafe fn fire_wait_event(
        con_in: *mut simple_text_input::Protocol,
        wait_for_event: BootWaitForEvent,
    ) -> io::Result<()> {
        let r = unsafe {
            let mut x: usize = 0;
            (wait_for_event)(1, [(*con_in).wait_for_key].as_mut_ptr(), &mut x)
        };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    unsafe fn read_key_stroke(con_in: *mut simple_text_input::Protocol) -> io::Result<u16> {
        let mut input_key = simple_text_input::InputKey::default();
        let r = unsafe { ((*con_in).read_key_stroke)(con_in, &mut input_key) };

        if r.is_error() {
            Err(status_to_io_error(r))
        } else if input_key.scan_code != 0 {
            Err(io::error::const_io_error!(io::ErrorKind::InvalidInput, "Invalid Input"))
        } else {
            Ok(input_key.unicode_char)
        }
    }

    unsafe fn reset_weak(con_in: *mut simple_text_input::Protocol) -> io::Result<()> {
        let r = unsafe { ((*con_in).reset)(con_in, r_efi::efi::Boolean::TRUE) };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    // Write a single Character to Stdout
    fn write_character(
        con_out: *mut simple_text_output::Protocol,
        character: ucs2::Ucs2Char,
    ) -> io::Result<()> {
        let mut buf: [u16; 2] = [character.into(), 0];
        let r = unsafe { ((*con_out).output_string)(con_out, buf.as_mut_ptr()) };

        if r.is_error() {
            Err(status_to_io_error(r))
        } else if character == ucs2::Ucs2Char::CR {
            // Handle enter key
            Self::write_character(con_out, ucs2::Ucs2Char::LF)
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
        let mut guid = uefi_command_protocol::PROTOCOL_GUID;
        if let Some(command_protocol) =
            common::get_current_handle_protocol::<uefi_command_protocol::Protocol>(&mut guid)
        {
            if let Some(pipe_protocol) =
                NonNull::new(unsafe { (*command_protocol.as_ptr()).stdout })
            {
                return pipe::AnonPipe::new(None, None, pipe_protocol).read(buf);
            }
        }
        let global_system_table =
            uefi::env::get_system_table().ok_or(common::SYSTEM_TABLE_ERROR)?;
        let con_in = get_con_in(global_system_table)?;
        let con_out = get_con_out(global_system_table)?;
        let wait_for_event = get_wait_for_event()?;

        if buf.len() < 3 {
            return Ok(0);
        }

        let ch = unsafe {
            Stdin::reset_weak(con_in.as_ptr())?;
            Stdin::fire_wait_event(con_in.as_ptr(), wait_for_event)?;
            Stdin::read_key_stroke(con_in.as_ptr())?
        };

        let ch = ucs2::Ucs2Char::from_u16(ch).ok_or(io::error::const_io_error!(
            io::ErrorKind::InvalidInput,
            "Invalid Character Input"
        ))?;
        Stdin::write_character(con_out.as_ptr(), ch)?;

        let ch = char::from(ch);
        let bytes_read = ch.len_utf8();

        // Replace CR with LF
        if ch == '\r' {
            '\n'.encode_utf8(buf);
        } else {
            ch.encode_utf8(buf);
        }

        Ok(bytes_read)
    }
}

impl Stdout {
    #[inline]
    pub const fn new() -> Stdout {
        Stdout(())
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut guid = uefi_command_protocol::PROTOCOL_GUID;
        if let Some(command_protocol) =
            common::get_current_handle_protocol::<uefi_command_protocol::Protocol>(&mut guid)
        {
            if let Some(pipe_protocol) =
                NonNull::new(unsafe { (*command_protocol.as_ptr()).stdout })
            {
                return pipe::AnonPipe::new(None, None, pipe_protocol).write(buf);
            }
        }

        let global_system_table =
            uefi::env::get_system_table().ok_or(common::SYSTEM_TABLE_ERROR)?;
        let con_out = get_con_out(global_system_table)?;
        unsafe { simple_text_output_write(con_out.as_ptr(), buf) }
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    #[inline]
    pub const fn new() -> Stderr {
        Stderr(())
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut guid = uefi_command_protocol::PROTOCOL_GUID;
        if let Some(command_protocol) =
            common::get_current_handle_protocol::<uefi_command_protocol::Protocol>(&mut guid)
        {
            if let Some(pipe_protocol) =
                NonNull::new(unsafe { (*command_protocol.as_ptr()).stderr })
            {
                return pipe::AnonPipe::new(None, None, pipe_protocol).write(buf);
            }
        }
        let global_system_table =
            uefi::env::get_system_table().ok_or(common::SYSTEM_TABLE_ERROR)?;
        let std_err = get_std_err(global_system_table)?;
        unsafe { simple_text_output_write(std_err.as_ptr(), buf) }
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[inline]
pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(r_efi::efi::Status::DEVICE_ERROR.as_usize() as i32)
}

#[inline]
pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}

fn utf8_to_ucs2(buf: &[u8], output: &mut [u16]) -> io::Result<usize> {
    let iter = ucs2::EncodeUcs2::from_bytes(buf).map_err(|_| {
        io::error::const_io_error!(io::ErrorKind::InvalidInput, "Invalid Output buffer")
    })?;
    let mut count = 0;
    let mut bytes_read = 0;

    for ch in iter {
        let c = match ch {
            Ok(x) => x,
            Err(_) => ucs2::Ucs2Char::REPLACEMENT_CHARACTER,
        };

        // Convert LF to CRLF
        if c == ucs2::Ucs2Char::LF {
            output[count] = u16::from(ucs2::Ucs2Char::CR);
            count += 1;

            if count + 1 >= output.len() {
                break;
            }
        }

        bytes_read += c.len_utf8();
        output[count] = u16::from(c);
        count += 1;

        if count + 1 >= output.len() {
            break;
        }
    }

    output[count] = 0;
    Ok(bytes_read)
}

// Write buffer to a Device supporting EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL
unsafe fn simple_text_output_write(
    protocol: *mut simple_text_output::Protocol,
    buf: &[u8],
) -> io::Result<usize> {
    let mut output = [0u16; MAX_BUFFER_SIZE / 2];
    let count = utf8_to_ucs2(buf, &mut output)?;
    let r = unsafe { ((*protocol).output_string)(protocol, output.as_mut_ptr()) };
    if r.is_error() { Err(status_to_io_error(r)) } else { Ok(count) }
}

// Returns error if `SystemTable->ConIn` is null.
#[inline]
fn get_con_in(
    st: NonNull<uefi::raw::SystemTable>,
) -> io::Result<NonNull<simple_text_input::Protocol>> {
    let con_in = unsafe { (*st.as_ptr()).con_in };
    NonNull::new(con_in).ok_or(io::error::const_io_error!(io::ErrorKind::NotFound, "ConIn"))
}

#[inline]
fn get_wait_for_event() -> io::Result<BootWaitForEvent> {
    let boot_services = uefi::env::get_boot_services().ok_or(common::BOOT_SERVICES_ERROR)?;
    Ok(unsafe { (*boot_services.as_ptr()).wait_for_event })
}

// Returns error if `SystemTable->ConOut` is null.
#[inline]
fn get_con_out(
    st: NonNull<uefi::raw::SystemTable>,
) -> io::Result<NonNull<simple_text_output::Protocol>> {
    let con_out = unsafe { (*st.as_ptr()).con_out };
    NonNull::new(con_out).ok_or(io::error::const_io_error!(io::ErrorKind::NotFound, "ConOut"))
}

// Returns error if `SystemTable->StdErr` is null.
#[inline]
fn get_std_err(
    st: NonNull<uefi::raw::SystemTable>,
) -> io::Result<NonNull<simple_text_output::Protocol>> {
    let std_err = unsafe { (*st.as_ptr()).std_err };
    NonNull::new(std_err).ok_or(io::error::const_io_error!(io::ErrorKind::NotFound, "StdErr"))
}
