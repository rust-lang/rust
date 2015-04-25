// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;
use io::prelude::*;
use os::windows::prelude::*;

use ffi::OsString;
use io::{self, Error, SeekFrom};
use libc::{self, HANDLE};
use mem;
use path::{Path, PathBuf};
use ptr;
use slice;
use sync::Arc;
use sys::handle::Handle;
use sys::{c, cvt};
use sys_common::FromInner;
use vec::Vec;

pub struct File { handle: Handle }
pub struct FileAttr { data: c::WIN32_FILE_ATTRIBUTE_DATA }

pub struct ReadDir {
    handle: FindNextFileHandle,
    root: Arc<PathBuf>,
    first: Option<libc::WIN32_FIND_DATAW>,
}

struct FindNextFileHandle(libc::HANDLE);

unsafe impl Send for FindNextFileHandle {}
unsafe impl Sync for FindNextFileHandle {}

pub struct DirEntry {
    root: Arc<PathBuf>,
    data: libc::WIN32_FIND_DATAW,
}

#[derive(Clone, Default)]
pub struct OpenOptions {
    create: bool,
    append: bool,
    read: bool,
    write: bool,
    truncate: bool,
    desired_access: Option<libc::DWORD>,
    share_mode: Option<libc::DWORD>,
    creation_disposition: Option<libc::DWORD>,
    flags_and_attributes: Option<libc::DWORD>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions { attrs: libc::DWORD }

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;
    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        if let Some(first) = self.first.take() {
            if let Some(e) = DirEntry::new(&self.root, &first) {
                return Some(Ok(e));
            }
        }
        unsafe {
            let mut wfd = mem::zeroed();
            loop {
                if libc::FindNextFileW(self.handle.0, &mut wfd) == 0 {
                    if libc::GetLastError() ==
                        c::ERROR_NO_MORE_FILES as libc::DWORD {
                        return None
                    } else {
                        return Some(Err(Error::last_os_error()))
                    }
                }
                if let Some(e) = DirEntry::new(&self.root, &wfd) {
                    return Some(Ok(e))
                }
            }
        }
    }
}

impl Drop for FindNextFileHandle {
    fn drop(&mut self) {
        let r = unsafe { libc::FindClose(self.0) };
        debug_assert!(r != 0);
    }
}

impl DirEntry {
    fn new(root: &Arc<PathBuf>, wfd: &libc::WIN32_FIND_DATAW) -> Option<DirEntry> {
        match &wfd.cFileName[0..3] {
            // check for '.' and '..'
            [46, 0, ..] |
            [46, 46, 0, ..] => return None,
            _ => {}
        }

        Some(DirEntry {
            root: root.clone(),
            data: *wfd,
        })
    }

    pub fn path(&self) -> PathBuf {
        let filename = super::truncate_utf16_at_nul(&self.data.cFileName);
        self.root.join(&<OsString as OsStringExt>::from_wide(filename))
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions { Default::default() }
    pub fn read(&mut self, read: bool) { self.read = read; }
    pub fn write(&mut self, write: bool) { self.write = write; }
    pub fn append(&mut self, append: bool) { self.append = append; }
    pub fn create(&mut self, create: bool) { self.create = create; }
    pub fn truncate(&mut self, truncate: bool) { self.truncate = truncate; }
    pub fn creation_disposition(&mut self, val: i32) {
        self.creation_disposition = Some(val as libc::DWORD);
    }
    pub fn flags_and_attributes(&mut self, val: i32) {
        self.flags_and_attributes = Some(val as libc::DWORD);
    }
    pub fn desired_access(&mut self, val: i32) {
        self.desired_access = Some(val as libc::DWORD);
    }
    pub fn share_mode(&mut self, val: i32) {
        self.share_mode = Some(val as libc::DWORD);
    }

    fn get_desired_access(&self) -> libc::DWORD {
        self.desired_access.unwrap_or({
            let mut base = if self.read {libc::FILE_GENERIC_READ} else {0} |
                           if self.write {libc::FILE_GENERIC_WRITE} else {0};
            if self.append {
                base &= !libc::FILE_WRITE_DATA;
                base |= libc::FILE_APPEND_DATA;
            }
            base
        })
    }

    fn get_share_mode(&self) -> libc::DWORD {
        // libuv has a good comment about this, but the basic idea is that
        // we try to emulate unix semantics by enabling all sharing by
        // allowing things such as deleting a file while it's still open.
        self.share_mode.unwrap_or(libc::FILE_SHARE_READ |
                                  libc::FILE_SHARE_WRITE |
                                  libc::FILE_SHARE_DELETE)
    }

    fn get_creation_disposition(&self) -> libc::DWORD {
        self.creation_disposition.unwrap_or({
            match (self.create, self.truncate) {
                (true, true) => libc::CREATE_ALWAYS,
                (true, false) => libc::OPEN_ALWAYS,
                (false, false) => libc::OPEN_EXISTING,
                (false, true) => {
                    if self.write && !self.append {
                        libc::CREATE_ALWAYS
                    } else {
                        libc::TRUNCATE_EXISTING
                    }
                }
            }
        })
    }

    fn get_flags_and_attributes(&self) -> libc::DWORD {
        self.flags_and_attributes.unwrap_or(libc::FILE_ATTRIBUTE_NORMAL)
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path = to_utf16(path);
        let handle = unsafe {
            libc::CreateFileW(path.as_ptr(),
                              opts.get_desired_access(),
                              opts.get_share_mode(),
                              ptr::null_mut(),
                              opts.get_creation_disposition(),
                              opts.get_flags_and_attributes(),
                              ptr::null_mut())
        };
        if handle == libc::INVALID_HANDLE_VALUE {
            Err(Error::last_os_error())
        } else {
            Ok(File { handle: Handle::new(handle) })
        }
    }

    pub fn fsync(&self) -> io::Result<()> {
        try!(cvt(unsafe { libc::FlushFileBuffers(self.handle.raw()) }));
        Ok(())
    }

    pub fn datasync(&self) -> io::Result<()> { self.fsync() }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        let mut info = c::FILE_END_OF_FILE_INFO {
            EndOfFile: size as libc::LARGE_INTEGER,
        };
        let size = mem::size_of_val(&info);
        try!(cvt(unsafe {
            c::SetFileInformationByHandle(self.handle.raw(),
                                          c::FileEndOfFileInfo,
                                          &mut info as *mut _ as *mut _,
                                          size as libc::DWORD)
        }));
        Ok(())
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        unsafe {
            let mut info: c::BY_HANDLE_FILE_INFORMATION = mem::zeroed();
            try!(cvt(c::GetFileInformationByHandle(self.handle.raw(),
                                                   &mut info)));
            Ok(FileAttr {
                data: c::WIN32_FILE_ATTRIBUTE_DATA {
                    dwFileAttributes: info.dwFileAttributes,
                    ftCreationTime: info.ftCreationTime,
                    ftLastAccessTime: info.ftLastAccessTime,
                    ftLastWriteTime: info.ftLastWriteTime,
                    nFileSizeHigh: info.nFileSizeHigh,
                    nFileSizeLow: info.nFileSizeLow,
                }
            })
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.handle.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.handle.write(buf)
    }

    pub fn flush(&self) -> io::Result<()> { Ok(()) }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            SeekFrom::Start(n) => (libc::FILE_BEGIN, n as i64),
            SeekFrom::End(n) => (libc::FILE_END, n),
            SeekFrom::Current(n) => (libc::FILE_CURRENT, n),
        };
        let pos = pos as libc::LARGE_INTEGER;
        let mut newpos = 0;
        try!(cvt(unsafe {
            libc::SetFilePointerEx(self.handle.raw(), pos,
                                   &mut newpos, whence)
        }));
        Ok(newpos as u64)
    }

    pub fn handle(&self) -> &Handle { &self.handle }
}

impl FromInner<libc::HANDLE> for File {
    fn from_inner(handle: libc::HANDLE) -> File {
        File { handle: Handle::new(handle) }
    }
}

pub fn to_utf16(s: &Path) -> Vec<u16> {
    s.as_os_str().encode_wide().chain(Some(0).into_iter()).collect()
}

impl FileAttr {
    pub fn is_dir(&self) -> bool {
        self.data.dwFileAttributes & c::FILE_ATTRIBUTE_DIRECTORY != 0
    }
    pub fn is_file(&self) -> bool {
        !self.is_dir()
    }
    pub fn size(&self) -> u64 {
        ((self.data.nFileSizeHigh as u64) << 32) | (self.data.nFileSizeLow as u64)
    }
    pub fn perm(&self) -> FilePermissions {
        FilePermissions { attrs: self.data.dwFileAttributes }
    }

    pub fn accessed(&self) -> u64 { self.to_ms(&self.data.ftLastAccessTime) }
    pub fn modified(&self) -> u64 { self.to_ms(&self.data.ftLastWriteTime) }

    fn to_ms(&self, ft: &libc::FILETIME) -> u64 {
        // FILETIME is in 100ns intervals and there are 10000 intervals in a
        // millisecond.
        let bits = (ft.dwLowDateTime as u64) | ((ft.dwHighDateTime as u64) << 32);
        bits / 10000
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.attrs & c::FILE_ATTRIBUTE_READONLY != 0
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.attrs |= c::FILE_ATTRIBUTE_READONLY;
        } else {
            self.attrs &= !c::FILE_ATTRIBUTE_READONLY;
        }
    }
}

pub fn mkdir(p: &Path) -> io::Result<()> {
    let p = to_utf16(p);
    try!(cvt(unsafe {
        libc::CreateDirectoryW(p.as_ptr(), ptr::null_mut())
    }));
    Ok(())
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let root = p.to_path_buf();
    let star = p.join("*");
    let path = to_utf16(&star);

    unsafe {
        let mut wfd = mem::zeroed();
        let find_handle = libc::FindFirstFileW(path.as_ptr(), &mut wfd);
        if find_handle != libc::INVALID_HANDLE_VALUE {
            Ok(ReadDir {
                handle: FindNextFileHandle(find_handle),
                root: Arc::new(root),
                first: Some(wfd),
            })
        } else {
            Err(Error::last_os_error())
        }
    }
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let p_utf16 = to_utf16(p);
    try!(cvt(unsafe { libc::DeleteFileW(p_utf16.as_ptr()) }));
    Ok(())
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let old = to_utf16(old);
    let new = to_utf16(new);
    try!(cvt(unsafe {
        libc::MoveFileExW(old.as_ptr(), new.as_ptr(),
                          libc::MOVEFILE_REPLACE_EXISTING)
    }));
    Ok(())
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let p = to_utf16(p);
    try!(cvt(unsafe { c::RemoveDirectoryW(p.as_ptr()) }));
    Ok(())
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    opts.flags_and_attributes(c::FILE_FLAG_OPEN_REPARSE_POINT as i32);
    let file = try!(File::open(p, &opts));

    let mut space = [0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
    let mut bytes = 0;

    unsafe {
        try!(cvt({
            c::DeviceIoControl(file.handle.raw(),
                               c::FSCTL_GET_REPARSE_POINT,
                               0 as *mut _,
                               0,
                               space.as_mut_ptr() as *mut _,
                               space.len() as libc::DWORD,
                               &mut bytes,
                               0 as *mut _)
        }));
        let buf: *const c::REPARSE_DATA_BUFFER = space.as_ptr() as *const _;
        if (*buf).ReparseTag != c::IO_REPARSE_TAG_SYMLINK {
            return Err(io::Error::new(io::ErrorKind::Other, "not a symlink"))
        }
        let info: *const c::SYMBOLIC_LINK_REPARSE_BUFFER =
                &(*buf).rest as *const _ as *const _;
        let path_buffer = &(*info).PathBuffer as *const _ as *const u16;
        let subst_off = (*info).SubstituteNameOffset / 2;
        let subst_ptr = path_buffer.offset(subst_off as isize);
        let subst_len = (*info).SubstituteNameLength / 2;
        let subst = slice::from_raw_parts(subst_ptr, subst_len as usize);

        Ok(PathBuf::from(OsString::from_wide(subst)))
    }

}

pub fn symlink(src: &Path, dst: &Path) -> io::Result<()> {
    use sys::c::compat::kernel32::CreateSymbolicLinkW;
    let src = to_utf16(src);
    let dst = to_utf16(dst);
    try!(cvt(unsafe {
        CreateSymbolicLinkW(dst.as_ptr(), src.as_ptr(), 0) as libc::BOOL
    }));
    Ok(())
}

pub fn link(src: &Path, dst: &Path) -> io::Result<()> {
    let src = to_utf16(src);
    let dst = to_utf16(dst);
    try!(cvt(unsafe {
        libc::CreateHardLinkW(dst.as_ptr(), src.as_ptr(), ptr::null_mut())
    }));
    Ok(())
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let p = to_utf16(p);
    unsafe {
        let mut attr: FileAttr = mem::zeroed();
        try!(cvt(c::GetFileAttributesExW(p.as_ptr(),
                                         c::GetFileExInfoStandard,
                                         &mut attr.data as *mut _ as *mut _)));
        Ok(attr)
    }
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    let p = to_utf16(p);
    unsafe {
        try!(cvt(c::SetFileAttributesW(p.as_ptr(), perm.attrs)));
        Ok(())
    }
}

pub fn utimes(p: &Path, atime: u64, mtime: u64) -> io::Result<()> {
    let atime = super::ms_to_filetime(atime);
    let mtime = super::ms_to_filetime(mtime);

    let mut o = OpenOptions::new();
    o.write(true);
    let f = try!(File::open(p, &o));
    try!(cvt(unsafe {
        c::SetFileTime(f.handle.raw(), 0 as *const _, &atime, &mtime)
    }));
    Ok(())
}
