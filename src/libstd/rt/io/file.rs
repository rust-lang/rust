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
use super::{SeekStyle,SeekSet, SeekCur, SeekEnd,
            Open, Read, Create, ReadWrite};
use rt::rtio::{RtioFileStream, IoFactory, IoFactoryObject};
use rt::io::{io_error, read_error, EndOfFile,
            FileMode, FileAccess, FileStat};
use rt::local::Local;
use option::{Some, None};
use path::Path;
use super::super::test::*;

/// Open a file for reading/writing, as indicated by `path`.
pub fn open<P: PathLike>(path: &P,
                         mode: FileMode,
                         access: FileAccess
                        ) -> Option<FileStream> {
    let open_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_open(path, mode, access)
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

/// Unlink (remove) a file from the filesystem, as indicated
/// by `path`.
pub fn unlink<P: PathLike>(path: &P) {
    let unlink_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_unlink(path)
    };
    match unlink_result {
        Ok(_) => (),
        Err(ioerr) => {
            io_error::cond.raise(ioerr);
        }
    }
}

/// Abstraction representing *positional* access to a file. In this case,
/// *positional* refers to it keeping an encounter *cursor* of where in the
/// file a subsequent `read` or `write` will begin from. Users of a `FileStream`
/// can `seek` to move the cursor to a given location *within the bounds of the
/// file* and can ask to have the `FileStream` `tell` them the location, in
/// bytes, of the cursor.
///
/// This abstraction is roughly modeled on the access workflow as represented
/// by `open(2)`, `read(2)`, `write(2)` and friends.
///
/// The `open` and `unlink` static methods are provided to manage creation/removal
/// of files. All other methods operatin on an instance of `FileStream`.
pub struct FileStream {
    fd: ~RtioFileStream,
    last_nread: int,
}

impl FileStream {
}

impl Reader for FileStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match self.fd.read(buf) {
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
        match self.fd.write(buf) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }

    fn flush(&mut self) {
        match self.fd.flush() {
            Ok(_) => (),
            Err(ioerr) => {
                read_error::cond.raise(ioerr);
            }
        }
    }
}

impl Seek for FileStream {
    fn tell(&self) -> u64 {
        let res = self.fd.tell();
        match res {
            Ok(cursor) => cursor,
            Err(ioerr) => {
                read_error::cond.raise(ioerr);
                return -1;
            }
        }
    }

    fn seek(&mut self, pos: i64, style: SeekStyle) {
        match self.fd.seek(pos, style) {
            Ok(_) => {
                // successful seek resets EOF indicator
                self.last_nread = -1;
                ()
            },
            Err(ioerr) => {
                read_error::cond.raise(ioerr);
            }
        }
    }
}

pub struct FileInfo(Path);

/// FIXME: DOCS
impl<'self> FileInfo {
    pub fn new<P: PathLike>(path: &P) -> FileInfo {
        do path.path_as_str |p| {
            FileInfo(Path(p))
        }
    }
    // FIXME #8873 can't put this in FileSystemInfo
    pub fn get_path(&'self self) -> &'self Path {
        &(**self)
    }
    pub fn stat(&self) -> Option<FileStat> {
        do io_error::cond.trap(|_| {
            // FIXME: can we do something more useful here?
        }).inside {
            stat(self.get_path())
        }
    }
    pub fn exists(&self) -> bool {
        match self.stat() {
            Some(s) => {
                match s.is_file {
                    true => {
                        true
                    },
                    false => {
                        // FIXME: raise condition?
                        false
                    }
                }
            },
            None => false
        }
    }
    pub fn is_file(&self) -> bool {
        match self.stat() {
            Some(s) => s.is_file,
            None => {
                // FIXME: raise condition
                false
            }
        }
    }
    pub fn open(&self, mode: FileMode, access: FileAccess) -> Option<FileStream> {
        match self.is_file() {
            true => {
                open(self.get_path(), mode, access)
            },
            false => {
                // FIXME: raise condition
                None
            }
        }
    }
    //fn open_read(&self) -> FileStream;
    //fn open_write(&self) -> FileStream;
    //fn create(&self) -> FileStream;
    //fn truncate(&self) -> FileStream;
    //fn open_or_create(&self) -> FileStream;
    //fn create_or_truncate(&self) -> FileStream;
    //fn unlink(&self);
}

/*
/// FIXME: DOCS
impl DirectoryInfo<'self> {
    fn new<P: PathLike>(path: &P) -> FileInfo {
        FileInfo(Path(path.path_as_str()))
    }
    // FIXME #8873 can't put this in FileSystemInfo
    fn get_path(&'self self) -> &'self Path {
        &*self
    }
    fn stat(&self) -> Option<FileStat> {
        file::stat(self.get_path())
    }
    fn exists(&self) -> bool {
        do io_error::cond.trap(|_| {
        }).inside {
            match self.stat() {
                Some(_) => true,
                None => false
            }
        }
    }
    fn is_dir(&self) -> bool {
        
    }
    fn create(&self);
    fn get_subdirs(&self, filter: &str) -> ~[Path];
    fn get_files(&self, filter: &str) -> ~[Path];
}
*/

/// Given a `rt::io::support::PathLike`, query the file system to get
/// information about a file, directory, etc.
///
/// Returns a `Some(PathInfo)` on success, and raises a `rt::io::IoError` condition
/// on failure and returns `None`.
pub fn stat<P: PathLike>(path: &P) -> Option<FileStat> {
    let open_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_stat(path)
    };
    match open_result {
        Ok(p) => {
            Some(p)
        },
        Err(ioerr) => {
            read_error::cond.raise(ioerr);
            None
        }
    }
}

fn file_test_smoke_test_impl() {
    do run_in_mt_newsched_task {
        let message = "it's alright. have a good time";
        let filename = &Path("./tmp/file_rt_io_file_test.txt");
        {
            let mut write_stream = open(filename, Create, ReadWrite).unwrap();
            write_stream.write(message.as_bytes());
        }
        {
            use str;
            let mut read_stream = open(filename, Open, Read).unwrap();
            let mut read_buf = [0, .. 1028];
            let read_str = match read_stream.read(read_buf).unwrap() {
                -1|0 => fail!("shouldn't happen"),
                n => str::from_utf8(read_buf.slice_to(n))
            };
            assert!(read_str == message.to_owned());
        }
        unlink(filename);
    }
}

#[test]
fn file_test_io_smoke_test() {
    file_test_smoke_test_impl();
}

fn file_test_invalid_path_opened_without_create_should_raise_condition_impl() {
    do run_in_mt_newsched_task {
        let filename = &Path("./tmp/file_that_does_not_exist.txt");
        let mut called = false;
        do io_error::cond.trap(|_| {
            called = true;
        }).inside {
            let result = open(filename, Open, Read);
            assert!(result.is_none());
        }
        assert!(called);
    }
}
#[test]
fn file_test_io_invalid_path_opened_without_create_should_raise_condition() {
    file_test_invalid_path_opened_without_create_should_raise_condition_impl();
}

fn file_test_unlinking_invalid_path_should_raise_condition_impl() {
    do run_in_mt_newsched_task {
        let filename = &Path("./tmp/file_another_file_that_does_not_exist.txt");
        let mut called = false;
        do io_error::cond.trap(|_| {
            called = true;
        }).inside {
            unlink(filename);
        }
        assert!(called);
    }
}
#[test]
fn file_test_iounlinking_invalid_path_should_raise_condition() {
    file_test_unlinking_invalid_path_should_raise_condition_impl();
}

fn file_test_io_non_positional_read_impl() {
    do run_in_mt_newsched_task {
        use str;
        let message = "ten-four";
        let mut read_mem = [0, .. 8];
        let filename = &Path("./tmp/file_rt_io_file_test_positional.txt");
        {
            let mut rw_stream = open(filename, Create, ReadWrite).unwrap();
            rw_stream.write(message.as_bytes());
        }
        {
            let mut read_stream = open(filename, Open, Read).unwrap();
            {
                let read_buf = read_mem.mut_slice(0, 4);
                read_stream.read(read_buf);
            }
            {
                let read_buf = read_mem.mut_slice(4, 8);
                read_stream.read(read_buf);
            }
        }
        unlink(filename);
        let read_str = str::from_utf8(read_mem);
        assert!(read_str == message.to_owned());
    }
}

#[test]
fn file_test_io_non_positional_read() {
    file_test_io_non_positional_read_impl();
}

fn file_test_io_seeking_impl() {
    do run_in_mt_newsched_task {
        use str;
        let message = "ten-four";
        let mut read_mem = [0, .. 4];
        let set_cursor = 4 as u64;
        let mut tell_pos_pre_read;
        let mut tell_pos_post_read;
        let filename = &Path("./tmp/file_rt_io_file_test_seeking.txt");
        {
            let mut rw_stream = open(filename, Create, ReadWrite).unwrap();
            rw_stream.write(message.as_bytes());
        }
        {
            let mut read_stream = open(filename, Open, Read).unwrap();
            read_stream.seek(set_cursor as i64, SeekSet);
            tell_pos_pre_read = read_stream.tell();
            read_stream.read(read_mem);
            tell_pos_post_read = read_stream.tell();
        }
        unlink(filename);
        let read_str = str::from_utf8(read_mem);
        assert!(read_str == message.slice(4, 8).to_owned());
        assert!(tell_pos_pre_read == set_cursor);
        assert!(tell_pos_post_read == message.len() as u64);
    }
}

#[test]
fn file_test_io_seek_and_tell_smoke_test() {
    file_test_io_seeking_impl();
}

fn file_test_io_seek_and_write_impl() {
    do run_in_mt_newsched_task {
        use str;
        let initial_msg =   "food-is-yummy";
        let overwrite_msg =    "-the-bar!!";
        let final_msg =     "foo-the-bar!!";
        let seek_idx = 3;
        let mut read_mem = [0, .. 13];
        let filename = &Path("./tmp/file_rt_io_file_test_seek_and_write.txt");
        {
            let mut rw_stream = open(filename, Create, ReadWrite).unwrap();
            rw_stream.write(initial_msg.as_bytes());
            rw_stream.seek(seek_idx as i64, SeekSet);
            rw_stream.write(overwrite_msg.as_bytes());
        }
        {
            let mut read_stream = open(filename, Open, Read).unwrap();
            read_stream.read(read_mem);
        }
        unlink(filename);
        let read_str = str::from_bytes(read_mem);
        assert!(read_str == final_msg.to_owned());
    }
}

#[test]
fn file_test_io_seek_and_write() {
    file_test_io_seek_and_write_impl();
}

fn file_test_io_seek_shakedown_impl() {
    do run_in_mt_newsched_task {
        use str;          // 01234567890123
        let initial_msg =   "qwer-asdf-zxcv";
        let chunk_one = "qwer";
        let chunk_two = "asdf";
        let chunk_three = "zxcv";
        let mut read_mem = [0, .. 4];
        let filename = &Path("./tmp/file_rt_io_file_test_seek_shakedown.txt");
        {
            let mut rw_stream = open(filename, Create, ReadWrite).unwrap();
            rw_stream.write(initial_msg.as_bytes());
        }
        {
            let mut read_stream = open(filename, Open, Read).unwrap();

            read_stream.seek(-4, SeekEnd);
            read_stream.read(read_mem);
            let read_str = str::from_utf8(read_mem);
            assert!(read_str == chunk_three.to_owned());

            read_stream.seek(-9, SeekCur);
            read_stream.read(read_mem);
            let read_str = str::from_utf8(read_mem);
            assert!(read_str == chunk_two.to_owned());

            read_stream.seek(0, SeekSet);
            read_stream.read(read_mem);
            let read_str = str::from_utf8(read_mem);
            assert!(read_str == chunk_one.to_owned());
        }
        unlink(filename);
    }
}

#[test]
fn file_test_io_seek_shakedown() {
    file_test_io_seek_shakedown_impl();
}

#[test]
fn file_test_stat_is_correct_on_is_file() {
    do run_in_newsched_task {
        let filename = &Path("./tmp/file_stat_correct_on_is_file.txt");
        {
            let mut fs = open(filename, Create, ReadWrite).unwrap();
            let msg = "hw";
            fs.write(msg.as_bytes());
        }
        let stat_res = match stat(filename) {
            Some(s) => s,
            None => fail!("shouldn't happen")
        };
        assert!(stat_res.is_file);
    }
}

#[test]
fn file_test_stat_is_correct_on_is_dir() {
    //assert!(false);
}

#[test]
fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
    //assert!(false);
}

#[test]
fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
    //assert!(false);
}
