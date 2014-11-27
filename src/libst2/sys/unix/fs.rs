// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Blocking posix-based file I/O

use libc::{mod, c_int, c_void};
use c_str::CString;
use mem;
use io;

use prelude::*;

use io::{FilePermission, Write, UnstableFileStat, Open, FileAccess, FileMode};
use io::{IoResult, FileStat, SeekStyle, Reader};
use io::{Read, Truncate, SeekCur, SeekSet, ReadWrite, SeekEnd, Append};
use result::{Ok, Err};
use sys::retry;
use sys_common::{keep_going, eof, mkerr_libc};

pub use path::PosixPath as Path;

pub type fd_t = libc::c_int;

pub struct FileDesc {
    /// The underlying C file descriptor.
    fd: fd_t,

    /// Whether to close the file descriptor on drop.
    close_on_drop: bool,
}

impl FileDesc {
    pub fn new(fd: fd_t, close_on_drop: bool) -> FileDesc { unimplemented!() }

    pub fn read(&self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
    pub fn write(&self, buf: &[u8]) -> IoResult<()> { unimplemented!() }

    pub fn fd(&self) -> fd_t { unimplemented!() }

    pub fn seek(&self, pos: i64, whence: SeekStyle) -> IoResult<u64> { unimplemented!() }

    pub fn tell(&self) -> IoResult<u64> { unimplemented!() }

    pub fn fsync(&self) -> IoResult<()> { unimplemented!() }

    pub fn datasync(&self) -> IoResult<()> { unimplemented!() }

    pub fn truncate(&self, offset: i64) -> IoResult<()> { unimplemented!() }

    pub fn fstat(&self) -> IoResult<FileStat> { unimplemented!() }

    /// Extract the actual filedescriptor without closing it.
    pub fn unwrap(self) -> fd_t { unimplemented!() }
}

impl Drop for FileDesc {
    fn drop(&mut self) { unimplemented!() }
}

pub fn open(path: &Path, fm: FileMode, fa: FileAccess) -> IoResult<FileDesc> { unimplemented!() }

pub fn mkdir(p: &Path, mode: uint) -> IoResult<()> { unimplemented!() }

pub fn readdir(p: &Path) -> IoResult<Vec<Path>> { unimplemented!() }

pub fn unlink(p: &Path) -> IoResult<()> { unimplemented!() }

pub fn rename(old: &Path, new: &Path) -> IoResult<()> { unimplemented!() }

pub fn chmod(p: &Path, mode: uint) -> IoResult<()> { unimplemented!() }

pub fn rmdir(p: &Path) -> IoResult<()> { unimplemented!() }

pub fn chown(p: &Path, uid: int, gid: int) -> IoResult<()> { unimplemented!() }

pub fn readlink(p: &Path) -> IoResult<Path> { unimplemented!() }

pub fn symlink(src: &Path, dst: &Path) -> IoResult<()> { unimplemented!() }

pub fn link(src: &Path, dst: &Path) -> IoResult<()> { unimplemented!() }

fn mkstat(stat: &libc::stat) -> FileStat { unimplemented!() }

pub fn stat(p: &Path) -> IoResult<FileStat> { unimplemented!() }

pub fn lstat(p: &Path) -> IoResult<FileStat> { unimplemented!() }

pub fn utime(p: &Path, atime: u64, mtime: u64) -> IoResult<()> { unimplemented!() }
