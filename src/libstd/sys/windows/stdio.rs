// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
use sys_common::io::read_to_end_uninitialized;

pub struct NoClose(Option<Handle>);

pub enum Output {
    Console(NoClose),
    Pipe(NoClose),
}

pub struct Stdin {
    handle: Output,
    utf8: Mutex<io::Cursor<Vec<u8>>>,
}
pub struct Stdout(Output);
pub struct Stderr(Output);

pub fn get(handle: c::DWORD) -> io::Result<Output> {
    let handle = unsafe { c::GetStdHandle(handle) };
    if handle == c::INVALID_HANDLE_VALUE {
        Err(io::Error::last_os_error())
    } else if handle.is_null() {
        Err(io::Error::new(io::ErrorKind::Other,
                           "no stdio handle available for this process"))
    } else {
        let ret = NoClose::new(handle);
        let mut out = 0;
        match unsafe { c::GetConsoleMode(handle, &mut out) } {
            0 => Ok(Output::Pipe(ret)),
            _ => Ok(Output::Console(ret)),
        }
    }
}

fn write(out: &Output, data: &[u8]) -> io::Result<usize> {
    let handle = match *out {
        Output::Console(ref c) => c.get().raw(),
        Output::Pipe(ref p) => return p.get().write(data),
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
        get(c::STD_INPUT_HANDLE).map(|handle| {
            Stdin {
                handle: handle,
                utf8: Mutex::new(Cursor::new(Vec::new())),
            }
        })
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let handle = match self.handle {
            Output::Console(ref c) => c.get().raw(),
            Output::Pipe(ref p) => return p.get().read(buf),
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
            if let Output::Console(_) = self.handle {
                if let Some(&last_byte) = data.last() {
                    if last_byte == CTRL_Z {
                        data.pop();
                    }
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

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        unsafe { read_to_end_uninitialized(self, buf) }
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> {
        get(c::STD_OUTPUT_HANDLE).map(Stdout)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        write(&self.0, data)
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> {
        get(c::STD_ERROR_HANDLE).map(Stderr)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        write(&self.0, data)
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

impl NoClose {
    fn new(handle: c::HANDLE) -> NoClose {
        NoClose(Some(Handle::new(handle)))
    }

    fn get(&self) -> &Handle { self.0.as_ref().unwrap() }
}

impl Drop for NoClose {
    fn drop(&mut self) {
        self.0.take().unwrap().into_raw();
    }
}

impl Output {
    pub fn handle(&self) -> &Handle {
        let nc = match *self {
            Output::Console(ref c) => c,
            Output::Pipe(ref c) => c,
        };
        nc.0.as_ref().unwrap()
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

pub const EBADF_ERR: i32 = ::sys::c::ERROR_INVALID_HANDLE as i32;
// The default buffer capacity is 64k, but apparently windows
// doesn't like 64k reads on stdin. See #13304 for details, but the
// idea is that on windows we use a slightly smaller buffer that's
// been seen to be acceptable.
pub const STDIN_BUF_SIZE: usize = 8 * 1024;
