// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use libc::{c_int,mode_t};
use ffi::OsString;
use path::{Path,PathBuf};
use sys::fd::FileDesc;
use sys::time::SystemTime;
use sys_common::FromInner;

#[derive(Debug)]
pub struct File(FileDesc);
#[derive(Clone)]
pub struct FileAttr(());
pub struct ReadDir(());
pub struct DirEntry(());
#[derive(Clone)]
pub struct OpenOptions(());
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions(());
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileType(());
pub struct DirBuilder(());

pub fn generic_error() -> io::Error {
    io::Error::new(io::ErrorKind::Other, "filesystem not supported on this platform")
}

impl FileAttr {
    pub fn size(&self) -> u64 { unimplemented!() }
    pub fn perm(&self) -> FilePermissions { unimplemented!() }
    pub fn file_type(&self) -> FileType { unimplemented!() }
    pub fn modified(&self) -> io::Result<SystemTime> { Err(generic_error()) }
    pub fn accessed(&self) -> io::Result<SystemTime> { Err(generic_error()) }
    pub fn created(&self) -> io::Result<SystemTime> { Err(generic_error()) }
}

/*
impl AsInner<stat64> for FileAttr {
    fn as_inner(&self) -> &stat64 { ... }
}
*/

impl FilePermissions {
    pub fn readonly(&self) -> bool { unimplemented!() }
    pub fn set_readonly(&mut self, _readonly: bool) { unimplemented!() }
    pub fn mode(&self) -> u32 { unimplemented!() }
}

impl FromInner<u32> for FilePermissions {
    fn from_inner(_mode: u32) -> FilePermissions {
        FilePermissions(())
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool { false }
    pub fn is_file(&self) -> bool { false }
    pub fn is_symlink(&self) -> bool { false }
    pub fn is(&self, _mode: mode_t) -> bool { false }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;
    fn next(&mut self) -> Option<io::Result<DirEntry>> { Some(Err(generic_error())) }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf { unimplemented!() }
    pub fn file_name(&self) -> OsString { unimplemented!() }
    pub fn metadata(&self) -> io::Result<FileAttr> { Err(generic_error()) }

    pub fn file_type(&self) -> io::Result<FileType> { Err(generic_error()) }

    pub fn ino(&self) -> u64 { unimplemented!() }
}

impl OpenOptions {
    pub fn new() -> OpenOptions { OpenOptions(()) }

    pub fn read(&mut self, _read: bool) {}
    pub fn write(&mut self, _write: bool) {}
    pub fn append(&mut self, _append: bool) {}
    pub fn truncate(&mut self, _truncate: bool) {}
    pub fn create(&mut self, _create: bool) {}
    pub fn create_new(&mut self, _create_new: bool) {}

    pub fn custom_flags(&mut self, _flags: i32) {}
    pub fn mode(&mut self, _mode: u32) {}
}

impl File {
    pub fn open(_path: &Path, _opts: &OpenOptions) -> io::Result<File> { Err(generic_error()) }
    pub fn file_attr(&self) -> io::Result<FileAttr> { Err(generic_error()) }
    pub fn fsync(&self) -> io::Result<()> { Err(generic_error()) }
    pub fn datasync(&self) -> io::Result<()> { Err(generic_error()) }
    pub fn truncate(&self, _size: u64) -> io::Result<()> { Err(generic_error()) }
    pub fn read(&self, _buf: &mut [u8]) -> io::Result<usize> { Err(generic_error()) }
    pub fn read_to_end(&self, _buf: &mut Vec<u8>) -> io::Result<usize> { Err(generic_error()) }
    pub fn write(&self, _buf: &[u8]) -> io::Result<usize> { Err(generic_error()) }
    pub fn flush(&self) -> io::Result<()> { Err(generic_error()) }
    pub fn seek(&self, _pos: io::SeekFrom) -> io::Result<u64> { Err(generic_error()) }
    pub fn duplicate(&self) -> io::Result<File> { Err(generic_error()) }

    pub fn fd(&self) -> &FileDesc { &self.0 }

    pub fn into_fd(self) -> FileDesc { self.0 }
}

impl FromInner<c_int> for File {
    fn from_inner(fd: c_int) -> File {
        File(FileDesc::new(fd))
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder { DirBuilder(()) }
    pub fn mkdir(&self, _p: &Path) -> io::Result<()> { Err(generic_error()) }
    pub fn set_mode(&mut self, _mode: u32) {}
}

pub fn readdir(_p: &Path) -> io::Result<ReadDir> { Err(generic_error()) }
pub fn unlink(_p: &Path) -> io::Result<()> { Err(generic_error()) }
pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> { Err(generic_error()) }
pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> { Err(generic_error()) }
pub fn rmdir(_p: &Path) -> io::Result<()> { Err(generic_error()) }
pub fn remove_dir_all(_path: &Path) -> io::Result<()> { Err(generic_error()) }
pub fn readlink(_p: &Path) -> io::Result<PathBuf> { Err(generic_error()) }
pub fn symlink(_src: &Path, _dst: &Path) -> io::Result<()> { Err(generic_error()) }
pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> { Err(generic_error()) }
pub fn stat(_p: &Path) -> io::Result<FileAttr> { Err(generic_error()) }
pub fn lstat(_p: &Path) -> io::Result<FileAttr> { Err(generic_error()) }
pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> { Err(generic_error()) }
pub fn copy(_from: &Path, _to: &Path) -> io::Result<u64> { Err(generic_error()) }

pub trait MetadataExt {
    fn st_dev(&self) -> u64;
    fn st_ino(&self) -> u64;
    fn st_mode(&self) -> u32;
    fn st_nlink(&self) -> u64;
    fn st_uid(&self) -> u32;
    fn st_gid(&self) -> u32;
    fn st_rdev(&self) -> u64;
    fn st_size(&self) -> u64;
    fn st_atime(&self) -> i64;
    fn st_atime_nsec(&self) -> i64;
    fn st_mtime(&self) -> i64;
    fn st_mtime_nsec(&self) -> i64;
    fn st_ctime(&self) -> i64;
    fn st_ctime_nsec(&self) -> i64;
    fn st_blksize(&self) -> u64;
    fn st_blocks(&self) -> u64;
}

impl MetadataExt for ::fs::Metadata {
    fn st_dev(&self) -> u64 { unimplemented!() }
    fn st_ino(&self) -> u64 { unimplemented!() }
    fn st_mode(&self) -> u32 { unimplemented!() }
    fn st_nlink(&self) -> u64 { unimplemented!() }
    fn st_uid(&self) -> u32 { unimplemented!() }
    fn st_gid(&self) -> u32 { unimplemented!() }
    fn st_rdev(&self) -> u64 { unimplemented!() }
    fn st_size(&self) -> u64 { unimplemented!() }
    fn st_atime(&self) -> i64 { unimplemented!() }
    fn st_atime_nsec(&self) -> i64 { unimplemented!() }
    fn st_mtime(&self) -> i64 { unimplemented!() }
    fn st_mtime_nsec(&self) -> i64 { unimplemented!() }
    fn st_ctime(&self) -> i64 { unimplemented!() }
    fn st_ctime_nsec(&self) -> i64 { unimplemented!() }
    fn st_blksize(&self) -> u64 { unimplemented!() }
    fn st_blocks(&self) -> u64 { unimplemented!() }
}
