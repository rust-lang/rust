// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Blocking posix-based file I/O

#[allow(non_camel_case_types)];

use libc;
use os;
use prelude::*;
use super::super::*;

fn raise_error() {
    // XXX: this should probably be a bit more descriptive...
    let (kind, desc) = match os::errno() as i32 {
        libc::EOF => (EndOfFile, "end of file"),
        _ => (OtherIoError, "unknown error"),
    };

    io_error::cond.raise(IoError {
        kind: kind,
        desc: desc,
        detail: Some(os::last_os_error())
    });
}

fn keep_going(data: &[u8], f: &fn(*u8, uint) -> i64) -> i64 {
    #[cfg(windows)] static eintr: int = 0; // doesn't matter
    #[cfg(not(windows))] static eintr: int = libc::EINTR as int;

    let (data, origamt) = do data.as_imm_buf |data, amt| { (data, amt) };
    let mut data = data;
    let mut amt = origamt;
    while amt > 0 {
        let mut ret;
        loop {
            ret = f(data, amt);
            if cfg!(not(windows)) { break } // windows has no eintr
            // if we get an eintr, then try again
            if ret != -1 || os::errno() as int != eintr { break }
        }
        if ret == 0 {
            break
        } else if ret != -1 {
            amt -= ret as uint;
            data = unsafe { data.offset(ret as int) };
        } else {
            return ret;
        }
    }
    return (origamt - amt) as i64;
}

pub type fd_t = libc::c_int;

pub struct FileDesc {
    priv fd: fd_t,
}

impl FileDesc {
    /// Create a `FileDesc` from an open C file descriptor.
    ///
    /// The `FileDesc` will take ownership of the specified file descriptor and
    /// close it upon destruction.
    ///
    /// Note that all I/O operations done on this object will be *blocking*, but
    /// they do not require the runtime to be active.
    pub fn new(fd: fd_t) -> FileDesc {
        FileDesc { fd: fd }
    }
}

impl Reader for FileDesc {
    #[fixed_stack_segment] #[inline(never)]
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        #[cfg(windows)] type rlen = libc::c_uint;
        #[cfg(not(windows))] type rlen = libc::size_t;
        let ret = do keep_going(buf) |buf, len| {
            unsafe {
                libc::read(self.fd, buf as *mut libc::c_void, len as rlen) as i64
            }
        };
        if ret == 0 {
            None
        } else if ret < 0 {
            raise_error();
            None
        } else {
            Some(ret as uint)
        }
    }

    fn eof(&mut self) -> bool { false }
}

impl Writer for FileDesc {
    #[fixed_stack_segment] #[inline(never)]
    fn write(&mut self, buf: &[u8]) {
        #[cfg(windows)] type wlen = libc::c_uint;
        #[cfg(not(windows))] type wlen = libc::size_t;
        let ret = do keep_going(buf) |buf, len| {
            unsafe {
                libc::write(self.fd, buf as *libc::c_void, len as wlen) as i64
            }
        };
        if ret < 0 {
            raise_error();
        }
    }

    fn flush(&mut self) {}
}

impl Drop for FileDesc {
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe { libc::close(self.fd); }
    }
}

pub struct CFile {
    priv file: *libc::FILE
}

impl CFile {
    /// Create a `CFile` from an open `FILE` pointer.
    ///
    /// The `CFile` takes ownership of the `FILE` pointer and will close it upon
    /// destruction.
    pub fn new(file: *libc::FILE) -> CFile { CFile { file: file } }
}

impl Reader for CFile {
    #[fixed_stack_segment] #[inline(never)]
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        let ret = do keep_going(buf) |buf, len| {
            unsafe {
                libc::fread(buf as *mut libc::c_void, 1, len as libc::size_t,
                            self.file) as i64
            }
        };
        if ret == 0 {
            None
        } else if ret < 0 {
            raise_error();
            None
        } else {
            Some(ret as uint)
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    fn eof(&mut self) -> bool {
        unsafe { libc::feof(self.file) != 0 }
    }
}

impl Writer for CFile {
    #[fixed_stack_segment] #[inline(never)]
    fn write(&mut self, buf: &[u8]) {
        let ret = do keep_going(buf) |buf, len| {
            unsafe {
                libc::fwrite(buf as *libc::c_void, 1, len as libc::size_t,
                            self.file) as i64
            }
        };
        if ret < 0 {
            raise_error();
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    fn flush(&mut self) {
        if unsafe { libc::fflush(self.file) } < 0 {
            raise_error();
        }
    }
}

impl Seek for CFile {
    #[fixed_stack_segment] #[inline(never)]
    fn tell(&self) -> u64 {
        let ret = unsafe { libc::ftell(self.file) };
        if ret < 0 {
            raise_error();
        }
        return ret as u64;
    }

    #[fixed_stack_segment] #[inline(never)]
    fn seek(&mut self, pos: i64, style: SeekStyle) {
        let whence = match style {
            SeekSet => libc::SEEK_SET,
            SeekEnd => libc::SEEK_END,
            SeekCur => libc::SEEK_CUR,
        };
        if unsafe { libc::fseek(self.file, pos as libc::c_long, whence) } < 0 {
            raise_error();
        }
    }
}

impl Drop for CFile {
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe { libc::fclose(self.file); }
    }
}

#[cfg(test)]
mod tests {
    use libc;
    use os;
    use prelude::*;
    use rt::io::{io_error, SeekSet};
    use super::*;

    #[test] #[fixed_stack_segment]
    #[ignore(cfg(target_os = "freebsd"))] // hmm, maybe pipes have a tiny buffer
    fn test_file_desc() {
        // Run this test with some pipes so we don't have to mess around with
        // opening or closing files.
        unsafe {
            let os::Pipe { input, out } = os::pipe();
            let mut reader = FileDesc::new(input);
            let mut writer = FileDesc::new(out);

            writer.write(bytes!("test"));
            let mut buf = [0u8, ..4];
            match reader.read(buf) {
                Some(4) => {
                    assert_eq!(buf[0], 't' as u8);
                    assert_eq!(buf[1], 'e' as u8);
                    assert_eq!(buf[2], 's' as u8);
                    assert_eq!(buf[3], 't' as u8);
                }
                r => fail2!("invalid read: {:?}", r)
            }

            let mut raised = false;
            do io_error::cond.trap(|_| { raised = true; }).inside {
                writer.read(buf);
            }
            assert!(raised);

            raised = false;
            do io_error::cond.trap(|_| { raised = true; }).inside {
                reader.write(buf);
            }
            assert!(raised);
        }
    }

    #[test] #[fixed_stack_segment]
    #[ignore(cfg(windows))] // apparently windows doesn't like tmpfile
    fn test_cfile() {
        unsafe {
            let f = libc::tmpfile();
            assert!(!f.is_null());
            let mut file = CFile::new(f);

            file.write(bytes!("test"));
            let mut buf = [0u8, ..4];
            file.seek(0, SeekSet);
            match file.read(buf) {
                Some(4) => {
                    assert_eq!(buf[0], 't' as u8);
                    assert_eq!(buf[1], 'e' as u8);
                    assert_eq!(buf[2], 's' as u8);
                    assert_eq!(buf[3], 't' as u8);
                }
                r => fail2!("invalid read: {:?}", r)
            }
        }
    }
}
