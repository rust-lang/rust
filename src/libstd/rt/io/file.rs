// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use super::support::PathLike;
use super::{Reader, Writer, Seek};
use super::SeekStyle;
use rt::rtio::{RtioFileDescriptor, IoFactory, IoFactoryObject};
use rt::io::{io_error, read_error, EndOfFile};
use rt::local::Local;
use rt::test::*;
use libc::{O_RDWR, O_RDONLY, O_WRONLY, S_IWUSR, S_IRUSR,
           O_CREAT, O_TRUNC, O_APPEND};

/// # FIXME #7785
/// * Ugh, this is ridiculous. What is the best way to represent these options?
enum FileMode {
    /// Opens an existing file. IoError if file does not exist.
    Open,
    /// Creates a file. IoError if file exists.
    Create,
    /// Opens an existing file or creates a new one.
    OpenOrCreate,
    /// Opens an existing file or creates a new one, positioned at EOF.
    Append,
    /// Opens an existing file, truncating it to 0 bytes.
    Truncate,
    /// Opens an existing file or creates a new one, truncating it to 0 bytes.
    CreateOrTruncate,
}

enum FileAccess {
    Read,
    Write,
    ReadWrite
}

pub struct FileStream {
    fd: ~RtioFileDescriptor,
    last_nread: int
}

impl FileStream {
    pub fn open<P: PathLike>(path: &P,
                             mode: FileMode,
                             access: FileAccess
                            ) -> Option<FileStream> {
        let open_result = unsafe {
            let io = Local::unsafe_borrow::<IoFactoryObject>();
            let mut flags = match mode {
                Open => 0,
                Create => O_CREAT,
                OpenOrCreate => O_CREAT,
                Append => O_APPEND,
                Truncate => O_TRUNC,
                CreateOrTruncate => O_TRUNC | O_CREAT
            };
            flags = match access {
                Read => flags | O_RDONLY,
                Write => flags | O_WRONLY,
                ReadWrite => flags | O_RDWR
            };
            let create_mode = match mode {
                Create|OpenOrCreate|CreateOrTruncate =>
                    S_IRUSR | S_IWUSR,
                _ => 0
            };
            (*io).fs_open(path, flags as int, create_mode as int)
        };
        match open_result {
            Ok(fd) => Some(FileStream {
                fd: fd,
                last_nread: -1
            }),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
    fn unlink<P: PathLike>(path: &P) {
        let unlink_result = unsafe {
            let io = Local::unsafe_borrow::<IoFactoryObject>();
            (*io).fs_unlink(path)
        };
        match unlink_result {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }
}

impl Reader for FileStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match self.fd.read(buf, 0) {
            Ok(read) => {
                self.last_nread = read;
                match read {
                    0 => None,
                    _ => Some(read as uint)
                }
            },
            Err(ioerr) => {
                // EOF is indicated by returning None
                if ioerr.kind != EndOfFile {
                    read_error::cond.raise(ioerr);
                }
                return None;
            }
        }
    }

    fn eof(&mut self) -> bool {
        self.last_nread == 0
    }
}

impl Writer for FileStream {
    fn write(&mut self, buf: &[u8]) {
        match self.fd.write(buf, 0) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }

    fn flush(&mut self) { fail!() }
}

impl Seek for FileStream {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}

fn file_test_smoke_test_impl() {
    do run_in_newsched_task {
        let message = "it's alright. have a good time";
        let filename = &Path("./rt_io_file_test.txt");
        {
            let mut write_stream = FileStream::open(filename, Create, ReadWrite).unwrap();
            write_stream.write(message.as_bytes());
        }
        {
            use str;
            let mut read_stream = FileStream::open(filename, Open, Read).unwrap();
            let mut read_buf = [0, .. 1028];
            let read_str = match read_stream.read(read_buf).unwrap() {
                -1|0 => fail!("shouldn't happen"),
                n => str::from_bytes(read_buf.slice_to(n))
            };
            assert!(read_str == message.to_owned());
        }
        FileStream::unlink(filename);
    }
}

#[test]
fn file_test_smoke_test() {
    file_test_smoke_test_impl();
}

fn file_test_invalid_path_opened_without_create_should_raise_condition_impl() {
    do run_in_newsched_task {
        let filename = &Path("./file_that_does_not_exist.txt");
        let mut called = false;
        do io_error::cond.trap(|_| {
            called = true;
        }).inside {
            let result = FileStream::open(filename, Open, Read);
            assert!(result.is_none());
        }
        assert!(called);
    }
}
#[test]
fn file_test_invalid_path_opened_without_create_should_raise_condition() {
    file_test_invalid_path_opened_without_create_should_raise_condition_impl();
}

fn file_test_unlinking_invalid_path_should_raise_condition_impl() {
    use io;
    do run_in_newsched_task {
        let filename = &Path("./another_file_that_does_not_exist.txt");
        let mut called = false;
        do io_error::cond.trap(|e| {
            io::println(fmt!("condition kind: %?", e.kind));
            called = true;
        }).inside {
            FileStream::unlink(filename);
        }
        assert!(called);
    }
}
#[test]
fn file_test_unlinking_invalid_path_should_raise_condition() {
    file_test_unlinking_invalid_path_should_raise_condition_impl();
}
