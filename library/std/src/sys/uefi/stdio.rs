use crate::io;
use crate::iter::Iterator;
use crate::mem::MaybeUninit;
use crate::os::uefi;
use crate::ptr::NonNull;

const MAX_BUFFER_SIZE: usize = 8192;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let st: NonNull<r_efi::efi::SystemTable> = uefi::env::system_table().cast();
        let stdin = unsafe { (*st.as_ptr()).con_in };

        // Try reading any pending data
        let inp = match read_key_stroke(stdin) {
            Ok(x) => x,
            Err(e) if e == r_efi::efi::Status::NOT_READY => {
                // Wait for keypress for new data
                wait_stdin(stdin)?;
                read_key_stroke(stdin).map_err(|x| io::Error::from_raw_os_error(x.as_usize()))?
            }
            Err(e) => {
                return Err(io::Error::from_raw_os_error(e.as_usize()));
            }
        };

        // Check if the key is printiable character
        if inp.scan_code != 0x00 {
            return Err(io::const_io_error!(io::ErrorKind::Interrupted, "Special Key Press"));
        }

        // SAFETY: Iterator will have only 1 character since we are reading only 1 Key
        // SAFETY: This character will always be UCS-2 and thus no surrogates.
        let ch: char = char::decode_utf16([inp.unicode_char]).next().unwrap().unwrap();
        if ch.len_utf8() > buf.len() {
            return Ok(0);
        }

        ch.encode_utf8(buf);

        Ok(ch.len_utf8())
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let st: NonNull<r_efi::efi::SystemTable> = uefi::env::system_table().cast();
        let stdout = unsafe { (*st.as_ptr()).con_out };

        write(stdout, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let st: NonNull<r_efi::efi::SystemTable> = uefi::env::system_table().cast();
        let stderr = unsafe { (*st.as_ptr()).std_err };

        write(stderr, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// UCS-2 character should occupy 3 bytes at most in UTF-8
pub const STDIN_BUF_SIZE: usize = 3;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<impl io::Write> {
    uefi::env::try_system_table().map(|_| Stderr::new())
}

fn write(
    protocol: *mut r_efi::protocols::simple_text_output::Protocol,
    buf: &[u8],
) -> io::Result<usize> {
    let mut utf16 = [0; MAX_BUFFER_SIZE / 2];

    // Get valid UTF-8 buffer
    let utf8 = match crate::str::from_utf8(buf) {
        Ok(x) => x,
        Err(e) => unsafe { crate::str::from_utf8_unchecked(&buf[..e.valid_up_to()]) },
    };
    // Clip UTF-8 buffer to max UTF-16 buffer we support
    let utf8 = &utf8[..utf8.floor_char_boundary(utf16.len() - 1)];

    for (i, ch) in utf8.encode_utf16().enumerate() {
        utf16[i] = ch;
    }

    unsafe { simple_text_output(protocol, &mut utf16) }?;

    Ok(utf8.len())
}

unsafe fn simple_text_output(
    protocol: *mut r_efi::protocols::simple_text_output::Protocol,
    buf: &mut [u16],
) -> io::Result<()> {
    let res = unsafe { ((*protocol).output_string)(protocol, buf.as_mut_ptr()) };
    if res.is_error() { Err(io::Error::from_raw_os_error(res.as_usize())) } else { Ok(()) }
}

fn wait_stdin(stdin: *mut r_efi::protocols::simple_text_input::Protocol) -> io::Result<()> {
    let boot_services: NonNull<r_efi::efi::BootServices> =
        uefi::env::boot_services().unwrap().cast();
    let wait_for_event = unsafe { (*boot_services.as_ptr()).wait_for_event };
    let wait_for_key_event = unsafe { (*stdin).wait_for_key };

    let r = {
        let mut x: usize = 0;
        (wait_for_event)(1, [wait_for_key_event].as_mut_ptr(), &mut x)
    };
    if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
}

fn read_key_stroke(
    stdin: *mut r_efi::protocols::simple_text_input::Protocol,
) -> Result<r_efi::protocols::simple_text_input::InputKey, r_efi::efi::Status> {
    let mut input_key: MaybeUninit<r_efi::protocols::simple_text_input::InputKey> =
        MaybeUninit::uninit();

    let r = unsafe { ((*stdin).read_key_stroke)(stdin, input_key.as_mut_ptr()) };

    if r.is_error() {
        Err(r)
    } else {
        let input_key = unsafe { input_key.assume_init() };
        Ok(input_key)
    }
}
