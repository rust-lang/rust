// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::prelude::*;
use os::windows::prelude::*;

use ffi::OsString;
use fmt;
use io::{self, Error, SeekFrom};
use mem;
use path::{Path, PathBuf};
use ptr;
use slice;
use sync::Arc;
use sys::handle::Handle;
use sys::time::SystemTime;
use sys::{c, cvt};
use sys_common::FromInner;
use vec::Vec;

use super::to_u16s;

pub struct File { handle: Handle }

#[derive(Clone)]
pub struct FileAttr {
    attributes: c::DWORD,
    creation_time: c::FILETIME,
    last_access_time: c::FILETIME,
    last_write_time: c::FILETIME,
    file_size: u64,
    reparse_tag: c::DWORD,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum FileType {
    Dir, File, SymlinkFile, SymlinkDir, ReparsePoint, MountPoint,
}

pub struct ReadDir {
    handle: FindNextFileHandle,
    root: Arc<PathBuf>,
    first: Option<c::WIN32_FIND_DATAW>,
}

struct FindNextFileHandle(c::HANDLE);

unsafe impl Send for FindNextFileHandle {}
unsafe impl Sync for FindNextFileHandle {}

pub struct DirEntry {
    root: Arc<PathBuf>,
    data: c::WIN32_FIND_DATAW,
}

#[derive(Clone)]
pub struct OpenOptions {
    // generic
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    // system-specific
    custom_flags: u32,
    access_mode: Option<c::DWORD>,
    attributes: c::DWORD,
    share_mode: c::DWORD,
    security_qos_flags: c::DWORD,
    security_attributes: usize, // FIXME: should be a reference
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct FilePermissions { readonly: bool }

pub struct DirBuilder;

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
                if c::FindNextFileW(self.handle.0, &mut wfd) == 0 {
                    if c::GetLastError() == c::ERROR_NO_MORE_FILES {
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
        let r = unsafe { c::FindClose(self.0) };
        debug_assert!(r != 0);
    }
}

impl DirEntry {
    fn new(root: &Arc<PathBuf>, wfd: &c::WIN32_FIND_DATAW) -> Option<DirEntry> {
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
        self.root.join(&self.file_name())
    }

    pub fn file_name(&self) -> OsString {
        let filename = super::truncate_utf16_at_nul(&self.data.cFileName);
        OsString::from_wide(filename)
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(FileType::new(self.data.dwFileAttributes,
                         /* reparse_tag = */ self.data.dwReserved0))
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        Ok(FileAttr {
            attributes: self.data.dwFileAttributes,
            creation_time: self.data.ftCreationTime,
            last_access_time: self.data.ftLastAccessTime,
            last_write_time: self.data.ftLastWriteTime,
            file_size: ((self.data.nFileSizeHigh as u64) << 32) | (self.data.nFileSizeLow as u64),
            reparse_tag: if self.data.dwFileAttributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                    // reserved unless this is a reparse point
                    self.data.dwReserved0
                } else {
                    0
                },
        })
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            // generic
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            // system-specific
            custom_flags: 0,
            access_mode: None,
            share_mode: c::FILE_SHARE_READ | c::FILE_SHARE_WRITE | c::FILE_SHARE_DELETE,
            attributes: 0,
            security_qos_flags: 0,
            security_attributes: 0,
        }
    }

    pub fn read(&mut self, read: bool) { self.read = read; }
    pub fn write(&mut self, write: bool) { self.write = write; }
    pub fn append(&mut self, append: bool) { self.append = append; }
    pub fn truncate(&mut self, truncate: bool) { self.truncate = truncate; }
    pub fn create(&mut self, create: bool) { self.create = create; }
    pub fn create_new(&mut self, create_new: bool) { self.create_new = create_new; }

    pub fn custom_flags(&mut self, flags: u32) { self.custom_flags = flags; }
    pub fn access_mode(&mut self, access_mode: u32) { self.access_mode = Some(access_mode); }
    pub fn share_mode(&mut self, share_mode: u32) { self.share_mode = share_mode; }
    pub fn attributes(&mut self, attrs: u32) { self.attributes = attrs; }
    pub fn security_qos_flags(&mut self, flags: u32) { self.security_qos_flags = flags; }
    pub fn security_attributes(&mut self, attrs: c::LPSECURITY_ATTRIBUTES) {
        self.security_attributes = attrs as usize;
    }

    fn get_access_mode(&self) -> io::Result<c::DWORD> {
        const ERROR_INVALID_PARAMETER: i32 = 87;

        match (self.read, self.write, self.append, self.access_mode) {
            (_, _, _, Some(mode)) => Ok(mode),
            (true,  false, false, None) => Ok(c::GENERIC_READ),
            (false, true,  false, None) => Ok(c::GENERIC_WRITE),
            (true,  true,  false, None) => Ok(c::GENERIC_READ | c::GENERIC_WRITE),
            (false, _,     true,  None) => Ok(c::FILE_GENERIC_WRITE & !c::FILE_WRITE_DATA),
            (true,  _,     true,  None) => Ok(c::GENERIC_READ |
                                              (c::FILE_GENERIC_WRITE & !c::FILE_WRITE_DATA)),
            (false, false, false, None) => Err(Error::from_raw_os_error(ERROR_INVALID_PARAMETER)),
        }
    }

    fn get_creation_mode(&self) -> io::Result<c::DWORD> {
        const ERROR_INVALID_PARAMETER: i32 = 87;

        match (self.write, self.append) {
            (true, false) => {}
            (false, false) =>
                if self.truncate || self.create || self.create_new {
                    return Err(Error::from_raw_os_error(ERROR_INVALID_PARAMETER));
                },
            (_, true) =>
                if self.truncate && !self.create_new {
                    return Err(Error::from_raw_os_error(ERROR_INVALID_PARAMETER));
                },
        }

        Ok(match (self.create, self.truncate, self.create_new) {
                (false, false, false) => c::OPEN_EXISTING,
                (true,  false, false) => c::OPEN_ALWAYS,
                (false, true,  false) => c::TRUNCATE_EXISTING,
                (true,  true,  false) => c::CREATE_ALWAYS,
                (_,      _,    true)  => c::CREATE_NEW,
           })
    }

    fn get_flags_and_attributes(&self) -> c::DWORD {
        self.custom_flags |
        self.attributes |
        self.security_qos_flags |
        if self.security_qos_flags != 0 { c::SECURITY_SQOS_PRESENT } else { 0 } |
        if self.create_new { c::FILE_FLAG_OPEN_REPARSE_POINT } else { 0 }
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path = try!(to_u16s(path));
        let handle = unsafe {
            c::CreateFileW(path.as_ptr(),
                           try!(opts.get_access_mode()),
                           opts.share_mode,
                           opts.security_attributes as *mut _,
                           try!(opts.get_creation_mode()),
                           opts.get_flags_and_attributes(),
                           ptr::null_mut())
        };
        if handle == c::INVALID_HANDLE_VALUE {
            Err(Error::last_os_error())
        } else {
            Ok(File { handle: Handle::new(handle) })
        }
    }

    pub fn fsync(&self) -> io::Result<()> {
        try!(cvt(unsafe { c::FlushFileBuffers(self.handle.raw()) }));
        Ok(())
    }

    pub fn datasync(&self) -> io::Result<()> { self.fsync() }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        let mut info = c::FILE_END_OF_FILE_INFO {
            EndOfFile: size as c::LARGE_INTEGER,
        };
        let size = mem::size_of_val(&info);
        try!(cvt(unsafe {
            c::SetFileInformationByHandle(self.handle.raw(),
                                          c::FileEndOfFileInfo,
                                          &mut info as *mut _ as *mut _,
                                          size as c::DWORD)
        }));
        Ok(())
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        unsafe {
            let mut info: c::BY_HANDLE_FILE_INFORMATION = mem::zeroed();
            try!(cvt(c::GetFileInformationByHandle(self.handle.raw(),
                                                   &mut info)));
            let mut attr = FileAttr {
                attributes: info.dwFileAttributes,
                creation_time: info.ftCreationTime,
                last_access_time: info.ftLastAccessTime,
                last_write_time: info.ftLastWriteTime,
                file_size: ((info.nFileSizeHigh as u64) << 32) | (info.nFileSizeLow as u64),
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

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.handle.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.handle.write(buf)
    }

    pub fn flush(&self) -> io::Result<()> { Ok(()) }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            SeekFrom::Start(n) => (c::FILE_BEGIN, n as i64),
            SeekFrom::End(n) => (c::FILE_END, n),
            SeekFrom::Current(n) => (c::FILE_CURRENT, n),
        };
        let pos = pos as c::LARGE_INTEGER;
        let mut newpos = 0;
        try!(cvt(unsafe {
            c::SetFilePointerEx(self.handle.raw(), pos,
                                &mut newpos, whence)
        }));
        Ok(newpos as u64)
    }

    pub fn rename(&self, new: &Path, replace: bool) -> io::Result<()> {
        // &self must be opened with DELETE permission
        use iter;
        #[cfg(target_arch = "x86")]
        const STRUCT_SIZE: usize = 12;
        #[cfg(target_arch = "x86_64")]
        const STRUCT_SIZE: usize = 20;

        // FIXME: check for internal NULs in 'new'
        let mut data: Vec<u16> = iter::repeat(0u16).take(STRUCT_SIZE/2)
                                 .chain(new.as_os_str().encode_wide())
                                 .collect();
        data.push(0);
        let size = data.len() * 2;

        unsafe {
            // Thanks to alignment guarantees on Windows this works
            // (8 for 32-bit and 16 for 64-bit)
            let mut info = data.as_mut_ptr() as *mut c::FILE_RENAME_INFO;
            // The type of ReplaceIfExists is BOOL, but it actually expects a
            // BOOLEAN. This means true is -1, not c::TRUE.
            (*info).ReplaceIfExists = if replace { -1 } else { c::FALSE };
            (*info).RootDirectory = ptr::null_mut();
            (*info).FileNameLength = (size - STRUCT_SIZE) as c::DWORD;
            try!(cvt(c::SetFileInformationByHandle(self.handle().raw(),
                                                   c::FileRenameInfo,
                                                   data.as_mut_ptr() as *mut _ as *mut _,
                                                   size as c::DWORD)));
            Ok(())
        }
    }

    pub fn set_attributes(&self, attr: c::DWORD) -> io::Result<()> {
        let mut info = c::FILE_BASIC_INFO {
            CreationTime: 0, // do not change
            LastAccessTime: 0, // do not change
            LastWriteTime: 0, // do not change
            ChangeTime: 0, // do not change
            FileAttributes: attr,
        };
        let size = mem::size_of_val(&info);
        try!(cvt(unsafe {
            c::SetFileInformationByHandle(self.handle.raw(),
                                          c::FileBasicInfo,
                                          &mut info as *mut _ as *mut _,
                                          size as c::DWORD)
        }));
        Ok(())
    }

    pub fn set_perm(&self, perm: FilePermissions) -> io::Result<()> {
        let attr = try!(self.file_attr()).attributes;
        if perm.readonly == (attr & c::FILE_ATTRIBUTE_READONLY != 0) {
            Ok(())
        } else if perm.readonly {
            self.set_attributes(attr | c::FILE_ATTRIBUTE_READONLY)
        } else {
            self.set_attributes(attr & !c::FILE_ATTRIBUTE_READONLY)
        }
    }

    pub fn duplicate(&self) -> io::Result<File> {
        Ok(File {
            handle: try!(self.handle.duplicate(0, true, c::DUPLICATE_SAME_ACCESS)),
        })
    }

    pub fn handle(&self) -> &Handle { &self.handle }

    pub fn into_handle(self) -> Handle { self.handle }

    fn reparse_point<'a>(&self,
                         space: &'a mut [u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE])
                         -> io::Result<(c::DWORD, &'a c::REPARSE_DATA_BUFFER)> {
        unsafe {
            let mut bytes = 0;
            try!(cvt({
                c::DeviceIoControl(self.handle.raw(),
                                   c::FSCTL_GET_REPARSE_POINT,
                                   ptr::null_mut(),
                                   0,
                                   space.as_mut_ptr() as *mut _,
                                   space.len() as c::DWORD,
                                   &mut bytes,
                                   ptr::null_mut())
            }));
            Ok((bytes, &*(space.as_ptr() as *const c::REPARSE_DATA_BUFFER)))
        }
    }

    fn readlink(&self) -> io::Result<PathBuf> {
        let mut space = [0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
        let (_bytes, buf) = try!(self.reparse_point(&mut space));
        unsafe {
            let (path_buffer, subst_off, subst_len, relative) = match buf.ReparseTag {
                c::IO_REPARSE_TAG_SYMLINK => {
                    let info: *const c::SYMBOLIC_LINK_REPARSE_BUFFER =
                        &buf.rest as *const _ as *const _;
                    (&(*info).PathBuffer as *const _ as *const u16,
                     (*info).SubstituteNameOffset / 2,
                     (*info).SubstituteNameLength / 2,
                     (*info).Flags & c::SYMLINK_FLAG_RELATIVE != 0)
                },
                c::IO_REPARSE_TAG_MOUNT_POINT => {
                    let info: *const c::MOUNT_POINT_REPARSE_BUFFER =
                        &buf.rest as *const _ as *const _;
                    (&(*info).PathBuffer as *const _ as *const u16,
                     (*info).SubstituteNameOffset / 2,
                     (*info).SubstituteNameLength / 2,
                     false)
                },
                _ => return Err(io::Error::new(io::ErrorKind::Other,
                                               "Unsupported reparse point type"))
            };
            let subst_ptr = path_buffer.offset(subst_off as isize);
            let mut subst = slice::from_raw_parts(subst_ptr, subst_len as usize);
            // Absolute paths start with an NT internal namespace prefix `\??\`
            // We should not let it leak through.
            if !relative && subst.starts_with(&[92u16, 63u16, 63u16, 92u16]) {
                subst = &subst[4..];
            }
            Ok(PathBuf::from(OsString::from_wide(subst)))
        }
    }
}

impl FromInner<c::HANDLE> for File {
    fn from_inner(handle: c::HANDLE) -> File {
        File { handle: Handle::new(handle) }
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME(#24570): add more info here (e.g. mode)
        let mut b = f.debug_struct("File");
        b.field("handle", &self.handle.raw());
        if let Ok(path) = get_path(&self) {
            b.field("path", &path);
        }
        b.finish()
    }
}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.file_size
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions {
            readonly: self.attributes & c::FILE_ATTRIBUTE_READONLY != 0
        }
    }

    pub fn attrs(&self) -> u32 { self.attributes as u32 }

    pub fn file_type(&self) -> FileType {
        FileType::new(self.attributes, self.reparse_tag)
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(self.last_write_time))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(self.last_access_time))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(self.creation_time))
    }

    pub fn modified_u64(&self) -> u64 {
        to_u64(&self.last_write_time)
    }

    pub fn accessed_u64(&self) -> u64 {
        to_u64(&self.last_access_time)
    }

    pub fn created_u64(&self) -> u64 {
        to_u64(&self.creation_time)
    }

    fn is_reparse_point(&self) -> bool {
        self.attributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
}

fn to_u64(ft: &c::FILETIME) -> u64 {
    (ft.dwLowDateTime as u64) | ((ft.dwHighDateTime as u64) << 32)
}

impl FilePermissions {
    pub fn new() -> FilePermissions { Default::default() }
    pub fn readonly(&self) -> bool { self.readonly }
    pub fn set_readonly(&mut self, readonly: bool) { self.readonly = readonly }
}

impl FileType {
    fn new(attrs: c::DWORD, reparse_tag: c::DWORD) -> FileType {
        match (attrs & c::FILE_ATTRIBUTE_DIRECTORY != 0,
               attrs & c::FILE_ATTRIBUTE_REPARSE_POINT != 0,
               reparse_tag) {
            (false, false, _) => FileType::File,
            (true, false, _) => FileType::Dir,
            (false, true, c::IO_REPARSE_TAG_SYMLINK) => FileType::SymlinkFile,
            (true, true, c::IO_REPARSE_TAG_SYMLINK) => FileType::SymlinkDir,
            (true, true, c::IO_REPARSE_TAG_MOUNT_POINT) => FileType::MountPoint,
            (_, true, _) => FileType::ReparsePoint,
            // Note: if a _file_ has a reparse tag of the type IO_REPARSE_TAG_MOUNT_POINT it is
            // invalid, as junctions always have to be dirs. We set the filetype to ReparsePoint
            // to indicate it is something symlink-like, but not something you can follow.
        }
    }

    pub fn is_dir(&self) -> bool { *self == FileType::Dir }
    pub fn is_file(&self) -> bool { *self == FileType::File }
    pub fn is_symlink(&self) -> bool {
        *self == FileType::SymlinkFile ||
        *self == FileType::SymlinkDir ||
        *self == FileType::MountPoint
    }
    pub fn is_symlink_dir(&self) -> bool {
        *self == FileType::SymlinkDir || *self == FileType::MountPoint
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder { DirBuilder }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        let p = try!(to_u16s(p));
        try!(cvt(unsafe {
            c::CreateDirectoryW(p.as_ptr(), ptr::null_mut())
        }));
        Ok(())
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let root = p.to_path_buf();
    let star = p.join("*");
    let path = try!(to_u16s(&star));

    unsafe {
        let mut wfd = mem::zeroed();
        let find_handle = c::FindFirstFileW(path.as_ptr(), &mut wfd);
        if find_handle != c::INVALID_HANDLE_VALUE {
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
    let p_u16s = try!(to_u16s(p));
    try!(cvt(unsafe { c::DeleteFileW(p_u16s.as_ptr()) }));
    Ok(())
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let old = try!(to_u16s(old));
    let new = try!(to_u16s(new));
    try!(cvt(unsafe {
        c::MoveFileExW(old.as_ptr(), new.as_ptr(), c::MOVEFILE_REPLACE_EXISTING)
    }));
    Ok(())
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let p = try!(to_u16s(p));
    try!(cvt(unsafe { c::RemoveDirectoryW(p.as_ptr()) }));
    Ok(())
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    // On Windows it is not enough to just recursively remove the contents of a
    // directory and then the directory itself. Deleting does not happen
    // instantaneously, but is scheduled.
    // To work around this, we move the file or directory to some `base_dir`
    // right before deletion to avoid races.
    //
    // As `base_dir` we choose the parent dir of the directory we want to
    // remove. We very probably have permission to create files here, as we
    // already need write permission in this dir to delete the directory. And it
    // should be on the same volume.
    //
    // To handle files with names like `CON` and `morse .. .`,  and when a
    // directory structure is so deep it needs long path names the path is first
    // converted to a `//?/`-path with `get_path()`.
    //
    // To make sure we don't leave a moved file laying around if the process
    // crashes before we can delete the file, we do all operations on an file
    // handle. By opening a file with `FILE_FLAG_DELETE_ON_CLOSE` Windows will
    // always delete the file when the handle closes.
    //
    // All files are renamed to be in the `base_dir`, and have their name
    // changed to "rm-<counter>". After every rename the counter is increased.
    // Rename should not overwrite possibly existing files in the base dir. So
    // if it fails with `AlreadyExists`, we just increase the counter and try
    // again.
    //
    // For read-only files and directories we first have to remove the read-only
    // attribute before we can move or delete them. This also removes the
    // attribute from possible hardlinks to the file, so just before closing we
    // restore the read-only attribute.
    //
    // If 'path' points to a directory symlink or junction we should not
    // recursively remove the target of the link, but only the link itself.
    //
    // Moving and deleting is guaranteed to succeed if we are able to open the
    // file with `DELETE` permission. If others have the file open we only have
    // `DELETE` permission if they have specified `FILE_SHARE_DELETE`. We can
    // also delete the file now, but it will not disappear until all others have
    // closed the file. But no-one can open the file after we have flagged it
    // for deletion.

    // Open the path once to get the canonical path, file type and attributes.
    let (path, metadata) = {
        let mut opts = OpenOptions::new();
        opts.access_mode(c::FILE_READ_ATTRIBUTES);
        opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS |
                          c::FILE_FLAG_OPEN_REPARSE_POINT);
        let file = try!(File::open(path, &opts));
        (try!(get_path(&file)), try!(file.file_attr()))
    };

    let mut ctx = RmdirContext {
        base_dir: match path.parent() {
            Some(dir) => dir,
            None => return Err(io::Error::new(io::ErrorKind::PermissionDenied,
                                              "can't delete root directory"))
        },
        readonly: metadata.perm().readonly(),
        counter: 0,
    };

    let filetype = metadata.file_type();
    if filetype.is_dir() {
        remove_dir_all_recursive(path.as_ref(), &mut ctx)
    } else if filetype.is_symlink_dir() {
        remove_item(path.as_ref(), &mut ctx)
    } else {
        Err(io::Error::new(io::ErrorKind::PermissionDenied, "Not a directory"))
    }
}

struct RmdirContext<'a> {
    base_dir: &'a Path,
    readonly: bool,
    counter: u64,
}

fn remove_dir_all_recursive(path: &Path, ctx: &mut RmdirContext)
                            -> io::Result<()> {
    let dir_readonly = ctx.readonly;
    for child in try!(readdir(path)) {
        let child = try!(child);
        let child_type = try!(child.file_type());
        ctx.readonly = try!(child.metadata()).perm().readonly();
        if child_type.is_dir() {
            try!(remove_dir_all_recursive(&child.path(), ctx));
        } else {
            try!(remove_item(&child.path().as_ref(), ctx));
        }
    }
    ctx.readonly = dir_readonly;
    remove_item(path, ctx)
}

fn remove_item(path: &Path, ctx: &mut RmdirContext) -> io::Result<()> {
    if !ctx.readonly {
        let mut opts = OpenOptions::new();
        opts.access_mode(c::DELETE);
        opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS | // delete directory
                          c::FILE_FLAG_OPEN_REPARSE_POINT | // delete symlink
                          c::FILE_FLAG_DELETE_ON_CLOSE);
        let file = try!(File::open(path, &opts));
        move_item(&file, ctx)
    } else {
        // remove read-only permision
        try!(set_perm(&path, FilePermissions::new()));
        // move and delete file, similar to !readonly.
        // only the access mode is different.
        let mut opts = OpenOptions::new();
        opts.access_mode(c::DELETE | c::FILE_WRITE_ATTRIBUTES);
        opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS |
                          c::FILE_FLAG_OPEN_REPARSE_POINT |
                          c::FILE_FLAG_DELETE_ON_CLOSE);
        let file = try!(File::open(path, &opts));
        try!(move_item(&file, ctx));
        // restore read-only flag just in case there are other hard links
        let mut perm = FilePermissions::new();
        perm.set_readonly(true);
        let _ = file.set_perm(perm); // ignore if this fails
        Ok(())
    }
}

fn move_item(file: &File, ctx: &mut RmdirContext) -> io::Result<()> {
    let mut tmpname = ctx.base_dir.join(format!{"rm-{}", ctx.counter});
    ctx.counter += 1;
    // Try to rename the file. If it already exists, just retry with an other
    // filename.
    while let Err(err) = file.rename(tmpname.as_ref(), false) {
        if err.kind() != io::ErrorKind::AlreadyExists { return Err(err) };
        tmpname = ctx.base_dir.join(format!("rm-{}", ctx.counter));
        ctx.counter += 1;
    }
    Ok(())
}

pub fn readlink(path: &Path) -> io::Result<PathBuf> {
    // Open the link with as few permissions as possible, instead of generic read.
    // By default FILE_LIST_DIRECTORY is denied for the junction "C:\Documents and Settings", so
    // this is needed for a common case.
    let mut opts = OpenOptions::new();
    opts.access_mode(c::FILE_READ_ATTRIBUTES);
    opts.custom_flags(c::FILE_FLAG_OPEN_REPARSE_POINT |
                      c::FILE_FLAG_BACKUP_SEMANTICS);
    let file = try!(File::open(&path, &opts));
    file.readlink()
}

pub fn symlink(src: &Path, dst: &Path) -> io::Result<()> {
    symlink_inner(src, dst, false)
}

pub fn symlink_inner(src: &Path, dst: &Path, dir: bool) -> io::Result<()> {
    let src = try!(to_u16s(src));
    let dst = try!(to_u16s(dst));
    let flags = if dir { c::SYMBOLIC_LINK_FLAG_DIRECTORY } else { 0 };
    try!(cvt(unsafe {
        c::CreateSymbolicLinkW(dst.as_ptr(), src.as_ptr(), flags) as c::BOOL
    }));
    Ok(())
}

pub fn link(src: &Path, dst: &Path) -> io::Result<()> {
    let src = try!(to_u16s(src));
    let dst = try!(to_u16s(dst));
    try!(cvt(unsafe {
        c::CreateHardLinkW(dst.as_ptr(), src.as_ptr(), ptr::null_mut())
    }));
    Ok(())
}

pub fn stat(path: &Path) -> io::Result<FileAttr> {
    let mut opts = OpenOptions::new();
    opts.access_mode(c::FILE_READ_ATTRIBUTES);
    // This flag is so we can open directories too
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS);
    let file = try!(File::open(path, &opts));
    file.file_attr()
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    let mut opts = OpenOptions::new();
    opts.access_mode(c::FILE_READ_ATTRIBUTES);
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS | c::FILE_FLAG_OPEN_REPARSE_POINT);
    let file = try!(File::open(path, &opts));
    file.file_attr()
}

pub fn set_perm(path: &Path, perm: FilePermissions) -> io::Result<()> {
    let mut opts = OpenOptions::new();
    opts.access_mode(c::FILE_READ_ATTRIBUTES | c::FILE_WRITE_ATTRIBUTES);
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS);
    let file = try!(File::open(path, &opts));
    file.set_perm(perm)
}

fn get_path(f: &File) -> io::Result<PathBuf> {
    super::fill_utf16_buf(|buf, sz| unsafe {
        c::GetFinalPathNameByHandleW(f.handle.raw(), buf, sz,
                                     c::VOLUME_NAME_DOS)
    }, |buf| {
        PathBuf::from(OsString::from_wide(buf))
    })
}

pub fn canonicalize(p: &Path) -> io::Result<PathBuf> {
    let mut opts = OpenOptions::new();
    // No read or write permissions are necessary
    opts.access_mode(0);
    // This flag is so we can open directories too
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS);
    let f = try!(File::open(p, &opts));
    get_path(&f)
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    unsafe extern "system" fn callback(
        _TotalFileSize: c::LARGE_INTEGER,
        TotalBytesTransferred: c::LARGE_INTEGER,
        _StreamSize: c::LARGE_INTEGER,
        _StreamBytesTransferred: c::LARGE_INTEGER,
        _dwStreamNumber: c::DWORD,
        _dwCallbackReason: c::DWORD,
        _hSourceFile: c::HANDLE,
        _hDestinationFile: c::HANDLE,
        lpData: c::LPVOID,
    ) -> c::DWORD {
        *(lpData as *mut i64) = TotalBytesTransferred;
        c::PROGRESS_CONTINUE
    }
    let pfrom = try!(to_u16s(from));
    let pto = try!(to_u16s(to));
    let mut size = 0i64;
    try!(cvt(unsafe {
        c::CopyFileExW(pfrom.as_ptr(), pto.as_ptr(), Some(callback),
                       &mut size as *mut _ as *mut _, ptr::null_mut(), 0)
    }));
    Ok(size as u64)
}

#[allow(dead_code)]
pub fn symlink_junction<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> io::Result<()> {
    symlink_junction_inner(src.as_ref(), dst.as_ref())
}

// Creating a directory junction on windows involves dealing with reparse
// points and the DeviceIoControl function, and this code is a skeleton of
// what can be found here:
//
// http://www.flexhex.com/docs/articles/hard-links.phtml
#[allow(dead_code)]
fn symlink_junction_inner(target: &Path, junction: &Path) -> io::Result<()> {
    let d = DirBuilder::new();
    try!(d.mkdir(&junction));

    let mut opts = OpenOptions::new();
    opts.write(true);
    opts.custom_flags(c::FILE_FLAG_OPEN_REPARSE_POINT |
                      c::FILE_FLAG_BACKUP_SEMANTICS);
    let f = try!(File::open(junction, &opts));
    let h = f.handle().raw();

    unsafe {
        let mut data = [0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
        let mut db = data.as_mut_ptr()
                        as *mut c::REPARSE_MOUNTPOINT_DATA_BUFFER;
        let buf = &mut (*db).ReparseTarget as *mut _;
        let mut i = 0;
        // FIXME: this conversion is very hacky
        let v = br"\??\";
        let v = v.iter().map(|x| *x as u16);
        for c in v.chain(target.as_os_str().encode_wide()) {
            *buf.offset(i) = c;
            i += 1;
        }
        *buf.offset(i) = 0;
        i += 1;
        (*db).ReparseTag = c::IO_REPARSE_TAG_MOUNT_POINT;
        (*db).ReparseTargetMaximumLength = (i * 2) as c::WORD;
        (*db).ReparseTargetLength = ((i - 1) * 2) as c::WORD;
        (*db).ReparseDataLength =
                (*db).ReparseTargetLength as c::DWORD + 12;

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

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use env;
    use fs::{self, File};
    use path::PathBuf;
    use rand::{self, Rng};
    use sys::c;

    macro_rules! check { ($e:expr) => (
        match $e {
            Ok(t) => t,
            Err(e) => panic!("{} failed with: {}", stringify!($e), e),
        }
    ) }

    macro_rules! error { ($e:expr, $s:expr) => (
        match $e {
            Ok(_) => panic!("Unexpected success. Should've been: {:?}", $s),
            Err(ref err) => assert!(err.to_string().contains($s),
                                    format!("`{}` did not contain `{}`", err, $s))
        }
    ) }

    pub struct TempDir(PathBuf);

    impl TempDir {
        fn join(&self, path: &str) -> PathBuf {
            let TempDir(ref p) = *self;
            p.join(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Gee, seeing how we're testing the fs module I sure hope that we
            // at least implement this correctly!
            let TempDir(ref p) = *self;
            check!(fs::remove_dir_all(p));
        }
    }

    pub fn tmpdir() -> TempDir {
        let p = env::temp_dir();
        let mut r = rand::thread_rng();
        let ret = p.join(&format!("rust-{}", r.next_u32()));
        check!(fs::create_dir(&ret));
        TempDir(ret)
    }

    #[test]
    fn set_perm_preserves_attr() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("file");
        check!(File::create(&path));

        let mut opts = super::OpenOptions::new();
        opts.read(true);
        opts.write(true);
        let file = check!(super::File::open(&path, &opts));

        let metadata = check!(file.file_attr());
        let attr = metadata.attributes;
        check!(file.set_attributes(attr | c::FILE_ATTRIBUTE_SYSTEM));

        let mut perm = metadata.perm();
        perm.set_readonly(true);
        check!(file.set_perm(perm));

        let new_metadata = check!(file.file_attr());
        assert!(new_metadata.attributes & c::FILE_ATTRIBUTE_SYSTEM != 0);
        assert!(new_metadata.perm().readonly());
    }
}
