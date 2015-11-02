// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use os::windows::prelude::*;

use borrow::Cow;
use ffi::{OsStr, OsString};
use fmt;
use io::SeekFrom;
use libc::{self, HANDLE};
use mem;
use ptr;
use slice;
use sync::Arc;
use sys::windows::handle::Handle;
use sys::windows::c::{self, cvt};
use sys::inner::*;
use sys::error::{self, Result, Error};
use vec::Vec;

pub struct File(Handle);
impl_inner!(File(Handle));

#[derive(Clone)]
pub struct FileAttr {
    data: c::WIN32_FILE_ATTRIBUTE_DATA,
    reparse_tag: libc::DWORD,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum FileType {
    Dir, File, Symlink, ReparsePoint, MountPoint,
}

pub struct ReadDir {
    handle: FindNextFileHandle,
    root: Arc<OsString>,
    first: Option<libc::WIN32_FIND_DATAW>,
}

struct FindNextFileHandle(libc::HANDLE);
impl_inner!(FindNextFileHandle(libc::HANDLE));

unsafe impl Send for FindNextFileHandle {}
unsafe impl Sync for FindNextFileHandle {}

pub struct DirEntry {
    root: Arc<OsString>,
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
    security_attributes: usize, // *mut T doesn't have a Default impl
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions { attrs: libc::DWORD }

pub struct DirBuilder;

impl Iterator for ReadDir {
    type Item = Result<DirEntry>;
    fn next(&mut self) -> Option<Result<DirEntry>> {
        if let Some(first) = self.first.take() {
            if let Some(e) = DirEntry::new(&self.root, &first) {
                return Some(Ok(e));
            }
        }
        unsafe {
            let mut wfd = mem::zeroed();
            loop {
                if libc::FindNextFileW(*self.handle.as_inner(), &mut wfd) == 0 {
                    if libc::GetLastError() ==
                        c::ERROR_NO_MORE_FILES as libc::DWORD {
                        return None
                    } else {
                        return Some(error::expect_last_result())
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
    fn new(root: &Arc<OsString>, wfd: &libc::WIN32_FIND_DATAW) -> Option<DirEntry> {
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

    pub fn root(&self) -> &OsStr {
        &self.root
    }

    pub fn file_name(&self) -> Cow<OsStr> {
        let filename = truncate_utf16_at_nul(&self.data.cFileName);
        Cow::Owned(OsString::from_wide(filename))
    }

    pub fn file_type(&self) -> Result<FileType> {
        Ok(FileType::new(self.data.dwFileAttributes,
                         /* reparse_tag = */ self.data.dwReserved0))
    }

    pub fn metadata(&self) -> Result<FileAttr> {
        Ok(FileAttr {
            data: c::WIN32_FILE_ATTRIBUTE_DATA {
                dwFileAttributes: self.data.dwFileAttributes,
                ftCreationTime: self.data.ftCreationTime,
                ftLastAccessTime: self.data.ftLastAccessTime,
                ftLastWriteTime: self.data.ftLastWriteTime,
                nFileSizeHigh: self.data.nFileSizeHigh,
                nFileSizeLow: self.data.nFileSizeLow,
            },
            reparse_tag: self.data.dwReserved0,
        })
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions { Default::default() }
    pub fn read(&mut self, read: bool) { self.read = read; }
    pub fn write(&mut self, write: bool) { self.write = write; }
    pub fn append(&mut self, append: bool) { self.append = append; }
    pub fn create(&mut self, create: bool) { self.create = create; }
    pub fn truncate(&mut self, truncate: bool) { self.truncate = truncate; }
    pub fn creation_disposition(&mut self, val: u32) {
        self.creation_disposition = Some(val);
    }
    pub fn flags_and_attributes(&mut self, val: u32) {
        self.flags_and_attributes = Some(val);
    }
    pub fn desired_access(&mut self, val: u32) {
        self.desired_access = Some(val);
    }
    pub fn share_mode(&mut self, val: u32) {
        self.share_mode = Some(val);
    }
    pub fn security_attributes(&mut self, attrs: libc::LPSECURITY_ATTRIBUTES) {
        self.security_attributes = attrs as usize;
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
    fn open_reparse_point(path: &OsStr, write: bool) -> Result<File> {
        let mut opts = OpenOptions::new();
        opts.read(!write);
        opts.write(write);
        opts.flags_and_attributes(c::FILE_FLAG_OPEN_REPARSE_POINT |
                                  c::FILE_FLAG_BACKUP_SEMANTICS);
        File::open(path, &opts)
    }

    pub fn open(path: &OsStr, opts: &OpenOptions) -> Result<File> {
        let path = to_utf16(path);
        let handle = unsafe {
            libc::CreateFileW(path.as_ptr(),
                              opts.get_desired_access(),
                              opts.get_share_mode(),
                              opts.security_attributes as *mut _,
                              opts.get_creation_disposition(),
                              opts.get_flags_and_attributes(),
                              ptr::null_mut())
        };
        if handle == libc::INVALID_HANDLE_VALUE {
            error::expect_last_result()
        } else {
            Ok(File(Handle::from_inner(handle)))
        }
    }

    pub fn fsync(&self) -> Result<()> {
        try!(cvt(unsafe { libc::FlushFileBuffers(*self.0.as_inner()) }));
        Ok(())
    }

    pub fn datasync(&self) -> Result<()> { self.fsync() }

    pub fn truncate(&self, size: u64) -> Result<()> {
        let mut info = c::FILE_END_OF_FILE_INFO {
            EndOfFile: size as libc::LARGE_INTEGER,
        };
        let size = mem::size_of_val(&info);
        try!(cvt(unsafe {
            c::SetFileInformationByHandle(*self.0.as_inner(),
                                          c::FileEndOfFileInfo,
                                          &mut info as *mut _ as *mut _,
                                          size as libc::DWORD)
        }));
        Ok(())
    }

    pub fn file_attr(&self) -> Result<FileAttr> {
        unsafe {
            let mut info: c::BY_HANDLE_FILE_INFORMATION = mem::zeroed();
            try!(cvt(c::GetFileInformationByHandle(*self.0.as_inner(),
                                                   &mut info)));
            let mut attr = FileAttr {
                data: c::WIN32_FILE_ATTRIBUTE_DATA {
                    dwFileAttributes: info.dwFileAttributes,
                    ftCreationTime: info.ftCreationTime,
                    ftLastAccessTime: info.ftLastAccessTime,
                    ftLastWriteTime: info.ftLastWriteTime,
                    nFileSizeHigh: info.nFileSizeHigh,
                    nFileSizeLow: info.nFileSizeLow,
                },
                reparse_tag: 0,
            };
            if attr.is_reparse_point() {
                let mut b = [0; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
                if let Ok((_, buf)) = self.reparse_point(&mut b) {
                    attr.reparse_tag = buf.ReparseTag;
                }
            }
            Ok(attr)
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.0.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        self.0.write(buf)
    }

    pub fn flush(&self) -> Result<()> { Ok(()) }

    pub fn seek(&self, pos: SeekFrom) -> Result<u64> {
        let (whence, pos) = match pos {
            SeekFrom::Start(n) => (libc::FILE_BEGIN, n as i64),
            SeekFrom::End(n) => (libc::FILE_END, n),
            SeekFrom::Current(n) => (libc::FILE_CURRENT, n),
        };
        let pos = pos as libc::LARGE_INTEGER;
        let mut newpos = 0;
        try!(cvt(unsafe {
            libc::SetFilePointerEx(*self.0.as_inner(), pos,
                                   &mut newpos, whence)
        }));
        Ok(newpos as u64)
    }

    fn reparse_point<'a>(&self,
                         space: &'a mut [u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE])
                         -> Result<(libc::DWORD, &'a c::REPARSE_DATA_BUFFER)> {
        unsafe {
            let mut bytes = 0;
            try!(cvt({
                c::DeviceIoControl(*self.0.as_inner(),
                                   c::FSCTL_GET_REPARSE_POINT,
                                   ptr::null_mut(),
                                   0,
                                   space.as_mut_ptr() as *mut _,
                                   space.len() as libc::DWORD,
                                   &mut bytes,
                                   ptr::null_mut())
            }));
            Ok((bytes, &*(space.as_ptr() as *const c::REPARSE_DATA_BUFFER)))
        }
    }

    fn readlink(&self) -> Result<OsString> {
        let mut space = [0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
        let (_bytes, buf) = try!(self.reparse_point(&mut space));
        if buf.ReparseTag != c::IO_REPARSE_TAG_SYMLINK {
            return Err(Error::NotSymlink)
        }

        unsafe {
            let info: *const c::SYMBOLIC_LINK_REPARSE_BUFFER =
                    &buf.rest as *const _ as *const _;
            let path_buffer = &(*info).PathBuffer as *const _ as *const u16;
            let subst_off = (*info).SubstituteNameOffset / 2;
            let subst_ptr = path_buffer.offset(subst_off as isize);
            let subst_len = (*info).SubstituteNameLength / 2;
            let subst = slice::from_raw_parts(subst_ptr, subst_len as usize);

            Ok(OsString::from(OsString::from_wide(subst)))
        }
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME(#24570): add more info here (e.g. mode)
        let mut b = f.debug_struct("File");
        b.field("handle", self.0.as_inner());
        if let Ok(path) = get_path(&self) {
            b.field("path", &path);
        }
        b.finish()
    }
}

pub fn to_utf16(s: &OsStr) -> Vec<u16> {
    s.encode_wide().chain(Some(0)).collect()
}

impl FileAttr {
    pub fn size(&self) -> u64 {
        ((self.data.nFileSizeHigh as u64) << 32) | (self.data.nFileSizeLow as u64)
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions { attrs: self.data.dwFileAttributes }
    }

    pub fn attrs(&self) -> u32 { self.data.dwFileAttributes as u32 }

    pub fn file_type(&self) -> FileType {
        FileType::new(self.data.dwFileAttributes, self.reparse_tag)
    }

    pub fn created(&self) -> u64 { self.to_u64(&self.data.ftCreationTime) }
    pub fn accessed(&self) -> u64 { self.to_u64(&self.data.ftLastAccessTime) }
    pub fn modified(&self) -> u64 { self.to_u64(&self.data.ftLastWriteTime) }

    fn to_u64(&self, ft: &libc::FILETIME) -> u64 {
        (ft.dwLowDateTime as u64) | ((ft.dwHighDateTime as u64) << 32)
    }

    fn is_reparse_point(&self) -> bool {
        self.data.dwFileAttributes & libc::FILE_ATTRIBUTE_REPARSE_POINT != 0
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

impl FileType {
    fn new(attrs: libc::DWORD, reparse_tag: libc::DWORD) -> FileType {
        if attrs & libc::FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            match reparse_tag {
                c::IO_REPARSE_TAG_SYMLINK => FileType::Symlink,
                c::IO_REPARSE_TAG_MOUNT_POINT => FileType::MountPoint,
                _ => FileType::ReparsePoint,
            }
        } else if attrs & c::FILE_ATTRIBUTE_DIRECTORY != 0 {
            FileType::Dir
        } else {
            FileType::File
        }
    }

    pub fn is_dir(&self) -> bool { *self == FileType::Dir }
    pub fn is_file(&self) -> bool { *self == FileType::File }
    pub fn is_symlink(&self) -> bool {
        *self == FileType::Symlink || *self == FileType::MountPoint
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder { DirBuilder }

    pub fn mkdir(&self, p: &OsStr) -> Result<()> {
        let p = to_utf16(p);
        try!(cvt(unsafe {
            libc::CreateDirectoryW(p.as_ptr(), ptr::null_mut())
        }));
        Ok(())
    }
}

pub fn readdir(p: &OsStr) -> Result<ReadDir> {
    use path::PathBuf;

    let root = PathBuf::from(p);
    let star = root.join("*");
    let path = to_utf16(star.as_os_str());

    unsafe {
        let mut wfd = mem::zeroed();
        let find_handle = libc::FindFirstFileW(path.as_ptr(), &mut wfd);
        if find_handle != libc::INVALID_HANDLE_VALUE {
            Ok(ReadDir {
                handle: FindNextFileHandle(find_handle),
                root: Arc::new(root.into()),
                first: Some(wfd),
            })
        } else {
            error::expect_last_result()
        }
    }
}

pub fn unlink(p: &OsStr) -> Result<()> {
    let p_utf16 = to_utf16(p);
    try!(cvt(unsafe { libc::DeleteFileW(p_utf16.as_ptr()) }));
    Ok(())
}

pub fn rename(old: &OsStr, new: &OsStr) -> Result<()> {
    let old = to_utf16(old);
    let new = to_utf16(new);
    try!(cvt(unsafe {
        libc::MoveFileExW(old.as_ptr(), new.as_ptr(),
                          libc::MOVEFILE_REPLACE_EXISTING)
    }));
    Ok(())
}

pub fn rmdir(p: &OsStr) -> Result<()> {
    let p = to_utf16(p);
    try!(cvt(unsafe { c::RemoveDirectoryW(p.as_ptr()) }));
    Ok(())
}

pub fn readlink(p: &OsStr) -> Result<OsString> {
    let file = try!(File::open_reparse_point(p, false));
    file.readlink()
}

pub fn symlink(src: &OsStr, dst: &OsStr) -> Result<()> {
    symlink_inner(src, dst, false)
}

pub fn symlink_inner(src: &OsStr, dst: &OsStr, dir: bool) -> Result<()> {
    let src = to_utf16(src);
    let dst = to_utf16(dst);
    let flags = if dir { c::SYMBOLIC_LINK_FLAG_DIRECTORY } else { 0 };
    try!(cvt(unsafe {
        c::CreateSymbolicLinkW(dst.as_ptr(), src.as_ptr(), flags) as libc::BOOL
    }));
    Ok(())
}

pub fn link(src: &OsStr, dst: &OsStr) -> Result<()> {
    let src = to_utf16(src);
    let dst = to_utf16(dst);
    try!(cvt(unsafe {
        libc::CreateHardLinkW(dst.as_ptr(), src.as_ptr(), ptr::null_mut())
    }));
    Ok(())
}

pub fn stat(p: &OsStr) -> Result<FileAttr> {
    let attr = try!(lstat(p));

    // If this is a reparse point, then we need to reopen the file to get the
    // actual destination. We also pass the FILE_FLAG_BACKUP_SEMANTICS flag to
    // ensure that we can open directories (this path may be a directory
    // junction). Once the file is opened we ask the opened handle what its
    // metadata information is.
    if attr.is_reparse_point() {
        let mut opts = OpenOptions::new();
        opts.flags_and_attributes(c::FILE_FLAG_BACKUP_SEMANTICS);
        let file = try!(File::open(p, &opts));
        file.file_attr()
    } else {
        Ok(attr)
    }
}

pub fn lstat(p: &OsStr) -> Result<FileAttr> {
    let utf16 = to_utf16(p);
    unsafe {
        let mut attr: FileAttr = mem::zeroed();
        try!(cvt(c::GetFileAttributesExW(utf16.as_ptr(),
                                         c::GetFileExInfoStandard,
                                         &mut attr.data as *mut _ as *mut _)));
        if attr.is_reparse_point() {
            attr.reparse_tag = File::open_reparse_point(p, false).and_then(|f| {
                let mut b = [0; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
                f.reparse_point(&mut b).map(|(_, b)| b.ReparseTag)
            }).unwrap_or(0);
        }
        Ok(attr)
    }
}

pub fn set_perm(p: &OsStr, perm: FilePermissions) -> Result<()> {
    let p = to_utf16(p);
    unsafe {
        try!(cvt(c::SetFileAttributesW(p.as_ptr(), perm.attrs)));
        Ok(())
    }
}

fn get_path(f: &File) -> Result<OsString> {
    c::fill_utf16_buf(|buf, sz| unsafe {
        c::GetFinalPathNameByHandleW(*f.0.as_inner(), buf, sz,
                                     libc::VOLUME_NAME_DOS)
    }, |buf| {
        OsString::from(OsString::from_wide(buf))
    })
}

pub fn canonicalize(p: &OsStr) -> Result<OsString> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    // This flag is so we can open directories too
    opts.flags_and_attributes(c::FILE_FLAG_BACKUP_SEMANTICS);
    let f = try!(File::open(p, &opts));
    get_path(&f)
}

pub const COPY_IMP: bool = true;

pub fn copy(from: &OsStr, to: &OsStr) -> Result<u64> {
    unsafe extern "system" fn callback(
        _TotalFileSize: libc::LARGE_INTEGER,
        TotalBytesTransferred: libc::LARGE_INTEGER,
        _StreamSize: libc::LARGE_INTEGER,
        _StreamBytesTransferred: libc::LARGE_INTEGER,
        _dwStreamNumber: libc::DWORD,
        _dwCallbackReason: libc::DWORD,
        _hSourceFile: HANDLE,
        _hDestinationFile: HANDLE,
        lpData: libc::LPVOID,
    ) -> libc::DWORD {
        *(lpData as *mut i64) = TotalBytesTransferred;
        c::PROGRESS_CONTINUE
    }
    let pfrom = to_utf16(from);
    let pto = to_utf16(to);
    let mut size = 0i64;
    try!(cvt(unsafe {
        c::CopyFileExW(pfrom.as_ptr(), pto.as_ptr(), Some(callback),
                       &mut size as *mut _ as *mut _, ptr::null_mut(), 0)
    }));
    Ok(size as u64)
}

#[test]
fn directory_junctions_are_directories() {
    use ffi::OsStr;
    use env;
    use rand::{self, StdRng, Rng};

    macro_rules! t {
        ($e:expr) => (match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with: {}", stringify!($e), e),
        })
    }

    let d = DirBuilder::new();
    let p = env::temp_dir();
    let mut r = rand::thread_rng();
    let ret = p.join(&format!("rust-{}", r.next_u32()));
    let foo = ret.join("foo");
    let bar = ret.join("bar");
    t!(d.mkdir(&ret));
    t!(d.mkdir(&foo));
    t!(d.mkdir(&bar));

    t!(create_junction(&bar, &foo));
    let metadata = stat(&bar);
    t!(delete_junction(&bar));

    t!(rmdir(&foo));
    t!(rmdir(&bar));
    t!(rmdir(&ret));

    let metadata = t!(metadata);
    assert!(metadata.file_type().is_dir());

    // Creating a directory junction on windows involves dealing with reparse
    // points and the DeviceIoControl function, and this code is a skeleton of
    // what can be found here:
    //
    // http://www.flexhex.com/docs/articles/hard-links.phtml
    fn create_junction(src: &OsStr, dst: &OsStr) -> Result<()> {
        let f = try!(opendir(src, true));
        let h = *f.as_inner().as_inner();

        unsafe {
            let mut data = [0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
            let mut db = data.as_mut_ptr()
                            as *mut c::REPARSE_MOUNTPOINT_DATA_BUFFER;
            let mut buf = &mut (*db).ReparseTarget as *mut _;
            let mut i = 0;
            let v = br"\??\";
            let v = v.iter().map(|x| *x as u16);
            for c in v.chain(dst.encode_wide()) {
                *buf.offset(i) = c;
                i += 1;
            }
            *buf.offset(i) = 0;
            i += 1;
            (*db).ReparseTag = c::IO_REPARSE_TAG_MOUNT_POINT;
            (*db).ReparseTargetMaximumLength = (i * 2) as libc::WORD;
            (*db).ReparseTargetLength = ((i - 1) * 2) as libc::WORD;
            (*db).ReparseDataLength =
                    (*db).ReparseTargetLength as libc::DWORD + 12;

            let mut ret = 0;
            cvt(c::DeviceIoControl(h as *mut _,
                                   c::FSCTL_SET_REPARSE_POINT,
                                   data.as_ptr() as *mut _,
                                   (*db).ReparseDataLength + 8,
                                   ptr::null_mut(), 0,
                                   &mut ret,
                                   ptr::null_mut())).map(|_| ())
        }
    }

    fn opendir(p: &OsStr, write: bool) -> Result<File> {
        unsafe {
            let mut token = ptr::null_mut();
            let mut tp: c::TOKEN_PRIVILEGES = mem::zeroed();
            try!(cvt(c::OpenProcessToken(c::GetCurrentProcess(),
                                         c::TOKEN_ADJUST_PRIVILEGES,
                                         &mut token)));
            let name: &OsStr = if write {
                "SeRestorePrivilege".as_ref()
            } else {
                "SeBackupPrivilege".as_ref()
            };
            let name = name.encode_wide().chain(Some(0)).collect::<Vec<_>>();
            try!(cvt(c::LookupPrivilegeValueW(ptr::null(),
                                              name.as_ptr(),
                                              &mut tp.Privileges[0].Luid)));
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Attributes = c::SE_PRIVILEGE_ENABLED;
            let size = mem::size_of::<c::TOKEN_PRIVILEGES>() as libc::DWORD;
            try!(cvt(c::AdjustTokenPrivileges(token, libc::FALSE, &mut tp, size,
                                              ptr::null_mut(), ptr::null_mut())));
            try!(cvt(libc::CloseHandle(token)));

            File::open_reparse_point(p, write)
        }
    }

    fn delete_junction(p: &OsStr) -> Result<()> {
        unsafe {
            let f = try!(opendir(p, true));
            let h = *f.as_inner().as_inner();
            let mut data = [0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
            let mut db = data.as_mut_ptr()
                            as *mut c::REPARSE_MOUNTPOINT_DATA_BUFFER;
            (*db).ReparseTag = c::IO_REPARSE_TAG_MOUNT_POINT;
            let mut bytes = 0;
            cvt(c::DeviceIoControl(h as *mut _,
                                   c::FSCTL_DELETE_REPARSE_POINT,
                                   data.as_ptr() as *mut _,
                                   (*db).ReparseDataLength + 8,
                                   ptr::null_mut(), 0,
                                   &mut bytes,
                                   ptr::null_mut())).map(|_| ())
        }
    }
}

fn truncate_utf16_at_nul<'a>(v: &'a [u16]) -> &'a [u16] {
    match v.iter().position(|c| *c == 0) {
        // don't include the 0
        Some(i) => &v[..i],
        None => v
    }
}
