// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;
use io::prelude::*;

use io::{self, Cursor};
use ptr;
use str;
use sync::Mutex;
use sys::c;
use sys::cvt;
use sys::handle::Handle;

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
    let utf16 = match str::from_utf8(data).ok() {
        Some(utf8) => utf8.utf16_units().collect::<Vec<u16>>(),
        None => return Err(invalid_encoding()),
    };
    let mut written = 0;
    try!(cvt(unsafe {
        c::WriteConsoleW(handle,
                         utf16.as_ptr() as c::LPCVOID,
                         utf16.len() as u32,
                         &mut written,
                         ptr::null_mut())
    }));

    // FIXME if this only partially writes the utf16 buffer then we need to
    //       figure out how many bytes of `data` were actually written
    assert_eq!(written as usize, utf16.len());
    Ok(data.len())
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
            try!(cvt(unsafe {
                c::ReadConsoleW(handle,
                                utf16.as_mut_ptr() as c::LPVOID,
                                utf16.len() as u32,
                                &mut num,
                                ptr::null_mut())
            }));
            utf16.truncate(num as usize);
            // FIXME: what to do about this data that has already been read?
            let data = match String::from_utf16(&utf16) {
                Ok(utf8) => utf8.into_bytes(),
                Err(..) => return Err(invalid_encoding()),
            };
            *utf8 = Cursor::new(data);
        }

        // MemReader shouldn't error here since we just filled it
        utf8.read(buf)
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> {
        get(c::STD_OUTPUT_HANDLE).map(Stdout)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        write(&self.0, data)
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> {
        get(c::STD_ERROR_HANDLE).map(Stderr)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        write(&self.0, data)
    }
}

// FIXME: right now this raw stderr handle is used in a few places because
//        std::io::stderr_raw isn't exposed, but once that's exposed this impl
//        should go away
impl io::Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Stderr::write(self, data)
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
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
