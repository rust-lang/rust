#![unstable(issue = "0", feature = "windows_stdio")]

use io::prelude::*;

use cmp;
use io::{self, Cursor};
use ptr;
use str;
use sync::Mutex;
use sys::c;
use sys::cvt;
use sys::handle::Handle;

pub enum Output {
    Console(c::HANDLE),
    Pipe(c::HANDLE),
}

pub struct Stdin {
    utf8: Mutex<io::Cursor<Vec<u8>>>,
}
pub struct Stdout;
pub struct Stderr;

pub fn get(handle: c::DWORD) -> io::Result<Output> {
    let handle = unsafe { c::GetStdHandle(handle) };
    if handle == c::INVALID_HANDLE_VALUE {
        Err(io::Error::last_os_error())
    } else if handle.is_null() {
        Err(io::Error::from_raw_os_error(c::ERROR_INVALID_HANDLE as i32))
    } else {
        let mut out = 0;
        match unsafe { c::GetConsoleMode(handle, &mut out) } {
            0 => Ok(Output::Pipe(handle)),
            _ => Ok(Output::Console(handle)),
        }
    }
}

fn write(handle: c::DWORD, data: &[u8]) -> io::Result<usize> {
    let handle = match get(handle)? {
        Output::Console(c) => c,
        Output::Pipe(p) => {
            let handle = Handle::new(p);
            let ret = handle.write(data);
            handle.into_raw();
            return ret
        }
    };

    // As with stdin on windows, stdout often can't handle writes of large
    // sizes. For an example, see #14940. For this reason, don't try to
    // write the entire output buffer on windows.
    //
    // For some other references, it appears that this problem has been
    // encountered by others [1] [2]. We choose the number 8K just because
    // libuv does the same.
    //
    // [1]: https://tahoe-lafs.org/trac/tahoe-lafs/ticket/1232
    // [2]: http://www.mail-archive.com/log4net-dev@logging.apache.org/msg00661.html
    const OUT_MAX: usize = 8192;
    let len = cmp::min(data.len(), OUT_MAX);
    let utf8 = match str::from_utf8(&data[..len]) {
        Ok(s) => s,
        Err(ref e) if e.valid_up_to() == 0 => return Err(invalid_encoding()),
        Err(e) => str::from_utf8(&data[..e.valid_up_to()]).unwrap(),
    };
    let utf16 = utf8.encode_utf16().collect::<Vec<u16>>();
    let mut written = 0;
    cvt(unsafe {
        c::WriteConsoleW(handle,
                         utf16.as_ptr() as c::LPCVOID,
                         utf16.len() as u32,
                         &mut written,
                         ptr::null_mut())
    })?;

    // FIXME if this only partially writes the utf16 buffer then we need to
    //       figure out how many bytes of `data` were actually written
    assert_eq!(written as usize, utf16.len());
    Ok(utf8.len())
}

impl Stdin {
    pub fn new() -> io::Result<Stdin> {
        Ok(Stdin {
            utf8: Mutex::new(Cursor::new(Vec::new())),
        })
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let handle = match get(c::STD_INPUT_HANDLE)? {
            Output::Console(c) => c,
            Output::Pipe(p) => {
                let handle = Handle::new(p);
                let ret = handle.read(buf);
                handle.into_raw();
                return ret
            }
        };
        let mut utf8 = self.utf8.lock().unwrap();
        // Read more if the buffer is empty
        if utf8.position() as usize == utf8.get_ref().len() {
            let mut utf16 = vec![0u16; 0x1000];
            let mut num = 0;
            let mut input_control = readconsole_input_control(CTRL_Z_MASK);
            cvt(unsafe {
                c::ReadConsoleW(handle,
                                utf16.as_mut_ptr() as c::LPVOID,
                                utf16.len() as u32,
                                &mut num,
                                &mut input_control as c::PCONSOLE_READCONSOLE_CONTROL)
            })?;
            utf16.truncate(num as usize);
            // FIXME: what to do about this data that has already been read?
            let mut data = match String::from_utf16(&utf16) {
                Ok(utf8) => utf8.into_bytes(),
                Err(..) => return Err(invalid_encoding()),
            };
            if let Some(&last_byte) = data.last() {
                if last_byte == CTRL_Z {
                    data.pop();
                }
            }
            *utf8 = Cursor::new(data);
        }

        // MemReader shouldn't error here since we just filled it
        utf8.read(buf)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }
}

#[unstable(reason = "not public", issue = "0", feature = "fd_read")]
impl<'a> Read for &'a Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> {
        Ok(Stdout)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        write(c::STD_OUTPUT_HANDLE, data)
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> {
        Ok(Stderr)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        write(c::STD_ERROR_HANDLE, data)
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

// FIXME: right now this raw stderr handle is used in a few places because
//        std::io::stderr_raw isn't exposed, but once that's exposed this impl
//        should go away
impl io::Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Stderr::write(self, data)
    }

    fn flush(&mut self) -> io::Result<()> {
        Stderr::flush(self)
    }
}

impl Output {
    pub fn handle(&self) -> c::HANDLE {
        match *self {
            Output::Console(c) => c,
            Output::Pipe(c) => c,
        }
    }
}

fn invalid_encoding() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "text was not valid unicode")
}

fn readconsole_input_control(wakeup_mask: c::ULONG) -> c::CONSOLE_READCONSOLE_CONTROL {
    c::CONSOLE_READCONSOLE_CONTROL {
        nLength: ::mem::size_of::<c::CONSOLE_READCONSOLE_CONTROL>() as c::ULONG,
        nInitialChars: 0,
        dwCtrlWakeupMask: wakeup_mask,
        dwControlKeyState: 0,
    }
}

const CTRL_Z: u8 = 0x1A;
const CTRL_Z_MASK: c::ULONG = 0x4000000; //1 << 0x1A

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(c::ERROR_INVALID_HANDLE as i32)
}

// The default buffer capacity is 64k, but apparently windows
// doesn't like 64k reads on stdin. See #13304 for details, but the
// idea is that on windows we use a slightly smaller buffer that's
// been seen to be acceptable.
pub const STDIN_BUF_SIZE: usize = 8 * 1024;

pub fn panic_output() -> Option<impl io::Write> {
    Stderr::new().ok()
}
