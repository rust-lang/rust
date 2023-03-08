#![allow(non_snake_case)]

use std::io;
use std::path::Path;

#[cfg(not(windows))]
pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    ::std::fs::remove_dir_all(path)
}

#[cfg(windows)]
pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    win::remove_dir_all(path)
}

#[cfg(windows)]
mod win {
    use winapi::ctypes::{c_uint, c_ushort};
    use winapi::shared::minwindef::{BOOL, DWORD, FALSE, FILETIME, LPVOID};
    use winapi::shared::winerror::{
        ERROR_CALL_NOT_IMPLEMENTED, ERROR_INSUFFICIENT_BUFFER, ERROR_NO_MORE_FILES,
    };
    use winapi::um::errhandlingapi::{GetLastError, SetLastError};
    use winapi::um::fileapi::{
        CreateFileW, FindFirstFileW, FindNextFileW, GetFileInformationByHandle,
    };
    use winapi::um::fileapi::{BY_HANDLE_FILE_INFORMATION, CREATE_ALWAYS, CREATE_NEW};
    use winapi::um::fileapi::{FILE_BASIC_INFO, FILE_RENAME_INFO, TRUNCATE_EXISTING};
    use winapi::um::fileapi::{OPEN_ALWAYS, OPEN_EXISTING};
    use winapi::um::handleapi::{CloseHandle, INVALID_HANDLE_VALUE};
    use winapi::um::ioapiset::DeviceIoControl;
    use winapi::um::libloaderapi::{GetModuleHandleW, GetProcAddress};
    use winapi::um::minwinbase::{
        FileBasicInfo, FileRenameInfo, FILE_INFO_BY_HANDLE_CLASS, WIN32_FIND_DATAW,
    };
    use winapi::um::winbase::SECURITY_SQOS_PRESENT;
    use winapi::um::winbase::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_DELETE_ON_CLOSE, FILE_FLAG_OPEN_REPARSE_POINT,
    };
    use winapi::um::winioctl::FSCTL_GET_REPARSE_POINT;
    use winapi::um::winnt::{DELETE, FILE_ATTRIBUTE_DIRECTORY, HANDLE, LPCWSTR};
    use winapi::um::winnt::{FILE_ATTRIBUTE_READONLY, FILE_ATTRIBUTE_REPARSE_POINT};
    use winapi::um::winnt::{FILE_GENERIC_WRITE, FILE_WRITE_DATA, GENERIC_READ, GENERIC_WRITE};
    use winapi::um::winnt::{FILE_READ_ATTRIBUTES, FILE_WRITE_ATTRIBUTES};
    use winapi::um::winnt::{FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE};
    use winapi::um::winnt::{IO_REPARSE_TAG_MOUNT_POINT, IO_REPARSE_TAG_SYMLINK, LARGE_INTEGER};

    use std::ffi::{OsStr, OsString};
    use std::io;
    use std::mem;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::path::{Path, PathBuf};
    use std::ptr;
    use std::sync::Arc;

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
        // To handle files with names like `CON` and `morse .. .`, and when a
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
            opts.access_mode(FILE_READ_ATTRIBUTES);
            opts.custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT);
            let file = File::open(path, &opts)?;
            (get_path(&file)?, file.file_attr()?)
        };

        let mut ctx = RmdirContext {
            base_dir: match path.parent() {
                Some(dir) => dir,
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "can't delete root directory",
                    ))
                }
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
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Not a directory",
            ))
        }
    }

    fn readdir(p: &Path) -> io::Result<ReadDir> {
        let root = p.to_path_buf();
        let star = p.join("*");
        let path = to_u16s(&star)?;

        unsafe {
            let mut wfd = mem::zeroed();
            let find_handle = FindFirstFileW(path.as_ptr(), &mut wfd);
            if find_handle != INVALID_HANDLE_VALUE {
                Ok(ReadDir {
                    handle: FindNextFileHandle(find_handle),
                    root: Arc::new(root),
                    first: Some(wfd),
                })
            } else {
                Err(io::Error::last_os_error())
            }
        }
    }

    struct RmdirContext<'a> {
        base_dir: &'a Path,
        readonly: bool,
        counter: u64,
    }

    fn remove_dir_all_recursive(path: &Path, ctx: &mut RmdirContext) -> io::Result<()> {
        let dir_readonly = ctx.readonly;
        for child in readdir(path)? {
            let child = child?;
            let child_type = child.file_type()?;
            ctx.readonly = child.metadata()?.perm().readonly();
            if child_type.is_dir() {
                remove_dir_all_recursive(&child.path(), ctx)?;
            } else {
                remove_item(&child.path().as_ref(), ctx)?;
            }
        }
        ctx.readonly = dir_readonly;
        remove_item(path, ctx)
    }

    fn remove_item(path: &Path, ctx: &mut RmdirContext) -> io::Result<()> {
        if !ctx.readonly {
            let mut opts = OpenOptions::new();
            opts.access_mode(DELETE);
            opts.custom_flags(
                FILE_FLAG_BACKUP_SEMANTICS | // delete directory
                              FILE_FLAG_OPEN_REPARSE_POINT | // delete symlink
                              FILE_FLAG_DELETE_ON_CLOSE,
            );
            let file = File::open(path, &opts)?;
            move_item(&file, ctx)
        } else {
            // remove read-only permision
            set_perm(&path, FilePermissions::new())?;
            // move and delete file, similar to !readonly.
            // only the access mode is different.
            let mut opts = OpenOptions::new();
            opts.access_mode(DELETE | FILE_WRITE_ATTRIBUTES);
            opts.custom_flags(
                FILE_FLAG_BACKUP_SEMANTICS
                    | FILE_FLAG_OPEN_REPARSE_POINT
                    | FILE_FLAG_DELETE_ON_CLOSE,
            );
            let file = File::open(path, &opts)?;
            move_item(&file, ctx)?;
            // restore read-only flag just in case there are other hard links
            let mut perm = FilePermissions::new();
            perm.set_readonly(true);
            let _ = file.set_perm(perm); // ignore if this fails
            Ok(())
        }
    }

    macro_rules! compat_fn {
        ($module:ident: $(
            fn $symbol:ident($($argname:ident: $argtype:ty),*)
                             -> $rettype:ty {
                $($body:expr);*
            }
        )*) => ($(
            #[allow(unused_variables)]
            unsafe fn $symbol($($argname: $argtype),*) -> $rettype {
                use std::sync::atomic::{AtomicUsize, Ordering};
                use std::mem;
                use std::ffi::CString;
                type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

                lazy_static! { static ref PTR: AtomicUsize = AtomicUsize::new(0);}

                fn lookup(module: &str, symbol: &str) -> Option<usize> {
                    let mut module: Vec<u16> = module.encode_utf16().collect();
                    module.push(0);
                    let symbol = CString::new(symbol).unwrap();
                    unsafe {
                        let handle = GetModuleHandleW(module.as_ptr());
                        match GetProcAddress(handle, symbol.as_ptr()) as usize {
                            0 => None,
                            n => Some(n),
                        }
                    }
                }

                fn store_func(ptr: &AtomicUsize, module: &str, symbol: &str,
                              fallback: usize) -> usize {
                    let value = lookup(module, symbol).unwrap_or(fallback);
                    ptr.store(value, Ordering::SeqCst);
                    value
                }

                fn load() -> usize {
                    store_func(&PTR, stringify!($module), stringify!($symbol), fallback as usize)
                }
                unsafe extern "system" fn fallback($($argname: $argtype),*)
                                                   -> $rettype {
                    $($body);*
                }

                let addr = match PTR.load(Ordering::SeqCst) {
                    0 => load(),
                    n => n,
                };
                mem::transmute::<usize, F>(addr)($($argname),*)
            }
        )*)
    }

    compat_fn! {
        kernel32:
        fn GetFinalPathNameByHandleW(_hFile: HANDLE,
                                     _lpszFilePath: LPCWSTR,
                                     _cchFilePath: DWORD,
                                     _dwFlags: DWORD) -> DWORD {
            SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0
        }
        fn SetFileInformationByHandle(_hFile: HANDLE,
                                      _FileInformationClass: FILE_INFO_BY_HANDLE_CLASS,
                                      _lpFileInformation: LPVOID,
                                      _dwBufferSize: DWORD) -> BOOL {
            SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0
        }
    }

    fn cvt(i: i32) -> io::Result<i32> {
        if i == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(i)
        }
    }

    fn to_u16s<S: AsRef<OsStr>>(s: S) -> io::Result<Vec<u16>> {
        fn inner(s: &OsStr) -> io::Result<Vec<u16>> {
            let mut maybe_result: Vec<u16> = s.encode_wide().collect();
            if maybe_result.iter().any(|&u| u == 0) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "strings passed to WinAPI cannot contain NULs",
                ));
            }
            maybe_result.push(0);
            Ok(maybe_result)
        }
        inner(s.as_ref())
    }

    fn truncate_utf16_at_nul<'a>(v: &'a [u16]) -> &'a [u16] {
        match v.iter().position(|c| *c == 0) {
            // don't include the 0
            Some(i) => &v[..i],
            None => v,
        }
    }

    fn fill_utf16_buf<F1, F2, T>(mut f1: F1, f2: F2) -> io::Result<T>
    where
        F1: FnMut(*mut u16, DWORD) -> DWORD,
        F2: FnOnce(&[u16]) -> T,
    {
        // Start off with a stack buf but then spill over to the heap if we end up
        // needing more space.
        let mut stack_buf = [0u16; 512];
        let mut heap_buf = Vec::new();
        unsafe {
            let mut n = stack_buf.len();
            loop {
                let buf = if n <= stack_buf.len() {
                    &mut stack_buf[..]
                } else {
                    let extra = n - heap_buf.len();
                    heap_buf.reserve(extra);
                    heap_buf.set_len(n);
                    &mut heap_buf[..]
                };

                // This function is typically called on windows API functions which
                // will return the correct length of the string, but these functions
                // also return the `0` on error. In some cases, however, the
                // returned "correct length" may actually be 0!
                //
                // To handle this case we call `SetLastError` to reset it to 0 and
                // then check it again if we get the "0 error value". If the "last
                // error" is still 0 then we interpret it as a 0 length buffer and
                // not an actual error.
                SetLastError(0);
                let k = match f1(buf.as_mut_ptr(), n as DWORD) {
                    0 if GetLastError() == 0 => 0,
                    0 => return Err(io::Error::last_os_error()),
                    n => n,
                } as usize;
                if k == n && GetLastError() == ERROR_INSUFFICIENT_BUFFER {
                    n *= 2;
                } else if k >= n {
                    n = k;
                } else {
                    return Ok(f2(&buf[..k]));
                }
            }
        }
    }

    #[derive(Clone, PartialEq, Eq, Debug, Default)]
    struct FilePermissions {
        readonly: bool,
    }

    impl FilePermissions {
        fn new() -> FilePermissions {
            Default::default()
        }
        fn readonly(&self) -> bool {
            self.readonly
        }
        fn set_readonly(&mut self, readonly: bool) {
            self.readonly = readonly
        }
    }

    #[derive(Clone)]
    struct OpenOptions {
        // generic
        read: bool,
        write: bool,
        append: bool,
        truncate: bool,
        create: bool,
        create_new: bool,
        // system-specific
        custom_flags: u32,
        access_mode: Option<DWORD>,
        attributes: DWORD,
        share_mode: DWORD,
        security_qos_flags: DWORD,
        security_attributes: usize, // FIXME: should be a reference
    }

    impl OpenOptions {
        fn new() -> OpenOptions {
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
                share_mode: FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                attributes: 0,
                security_qos_flags: 0,
                security_attributes: 0,
            }
        }
        fn custom_flags(&mut self, flags: u32) {
            self.custom_flags = flags;
        }
        fn access_mode(&mut self, access_mode: u32) {
            self.access_mode = Some(access_mode);
        }

        fn get_access_mode(&self) -> io::Result<DWORD> {
            const ERROR_INVALID_PARAMETER: i32 = 87;

            match (self.read, self.write, self.append, self.access_mode) {
                (_, _, _, Some(mode)) => Ok(mode),
                (true, false, false, None) => Ok(GENERIC_READ),
                (false, true, false, None) => Ok(GENERIC_WRITE),
                (true, true, false, None) => Ok(GENERIC_READ | GENERIC_WRITE),
                (false, _, true, None) => Ok(FILE_GENERIC_WRITE & !FILE_WRITE_DATA),
                (true, _, true, None) => Ok(GENERIC_READ | (FILE_GENERIC_WRITE & !FILE_WRITE_DATA)),
                (false, false, false, None) => {
                    Err(io::Error::from_raw_os_error(ERROR_INVALID_PARAMETER))
                }
            }
        }

        fn get_creation_mode(&self) -> io::Result<DWORD> {
            const ERROR_INVALID_PARAMETER: i32 = 87;

            match (self.write, self.append) {
                (true, false) => {}
                (false, false) => {
                    if self.truncate || self.create || self.create_new {
                        return Err(io::Error::from_raw_os_error(ERROR_INVALID_PARAMETER));
                    }
                }
                (_, true) => {
                    if self.truncate && !self.create_new {
                        return Err(io::Error::from_raw_os_error(ERROR_INVALID_PARAMETER));
                    }
                }
            }

            Ok(match (self.create, self.truncate, self.create_new) {
                (false, false, false) => OPEN_EXISTING,
                (true, false, false) => OPEN_ALWAYS,
                (false, true, false) => TRUNCATE_EXISTING,
                (true, true, false) => CREATE_ALWAYS,
                (_, _, true) => CREATE_NEW,
            })
        }

        fn get_flags_and_attributes(&self) -> DWORD {
            self.custom_flags
                | self.attributes
                | self.security_qos_flags
                | if self.security_qos_flags != 0 {
                    SECURITY_SQOS_PRESENT
                } else {
                    0
                }
                | if self.create_new {
                    FILE_FLAG_OPEN_REPARSE_POINT
                } else {
                    0
                }
        }
    }

    struct File {
        handle: Handle,
    }

    impl File {
        fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
            let path = to_u16s(path)?;
            let handle = unsafe {
                CreateFileW(
                    path.as_ptr(),
                    opts.get_access_mode()?,
                    opts.share_mode,
                    opts.security_attributes as *mut _,
                    opts.get_creation_mode()?,
                    opts.get_flags_and_attributes(),
                    ptr::null_mut(),
                )
            };
            if handle == INVALID_HANDLE_VALUE {
                Err(io::Error::last_os_error())
            } else {
                Ok(File {
                    handle: Handle::new(handle),
                })
            }
        }

        fn file_attr(&self) -> io::Result<FileAttr> {
            unsafe {
                let mut info: BY_HANDLE_FILE_INFORMATION = mem::zeroed();
                cvt(GetFileInformationByHandle(self.handle.raw(), &mut info))?;
                let mut attr = FileAttr {
                    attributes: info.dwFileAttributes,
                    creation_time: info.ftCreationTime,
                    last_access_time: info.ftLastAccessTime,
                    last_write_time: info.ftLastWriteTime,
                    file_size: ((info.nFileSizeHigh as u64) << 32) | (info.nFileSizeLow as u64),
                    reparse_tag: 0,
                };
                if attr.is_reparse_point() {
                    let mut b = [0; MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
                    if let Ok((_, buf)) = self.reparse_point(&mut b) {
                        attr.reparse_tag = buf.ReparseTag;
                    }
                }
                Ok(attr)
            }
        }

        fn set_attributes(&self, attr: DWORD) -> io::Result<()> {
            let zero: LARGE_INTEGER = unsafe { mem::zeroed() };

            let mut info = FILE_BASIC_INFO {
                CreationTime: zero,   // do not change
                LastAccessTime: zero, // do not change
                LastWriteTime: zero,  // do not change
                ChangeTime: zero,     // do not change
                FileAttributes: attr,
            };
            let size = mem::size_of_val(&info);
            cvt(unsafe {
                SetFileInformationByHandle(
                    self.handle.raw(),
                    FileBasicInfo,
                    &mut info as *mut _ as *mut _,
                    size as DWORD,
                )
            })?;
            Ok(())
        }

        fn rename(&self, new: &Path, replace: bool) -> io::Result<()> {
            // &self must be opened with DELETE permission
            use std::iter;
            #[cfg(target_arch = "x86")]
            const STRUCT_SIZE: usize = 12;
            #[cfg(target_arch = "x86_64")]
            const STRUCT_SIZE: usize = 20;

            // FIXME: check for internal NULs in 'new'
            let mut data: Vec<u16> = iter::repeat(0u16)
                .take(STRUCT_SIZE / 2)
                .chain(new.as_os_str().encode_wide())
                .collect();
            data.push(0);
            let size = data.len() * 2;

            unsafe {
                // Thanks to alignment guarantees on Windows this works
                // (8 for 32-bit and 16 for 64-bit)
                let info = data.as_mut_ptr() as *mut FILE_RENAME_INFO;
                // The type of ReplaceIfExists is BOOL, but it actually expects a
                // BOOLEAN. This means true is -1, not c::TRUE.
                (*info).ReplaceIfExists = if replace { -1 } else { FALSE };
                (*info).RootDirectory = ptr::null_mut();
                (*info).FileNameLength = (size - STRUCT_SIZE) as DWORD;
                cvt(SetFileInformationByHandle(
                    self.handle().raw(),
                    FileRenameInfo,
                    data.as_mut_ptr() as *mut _ as *mut _,
                    size as DWORD,
                ))?;
                Ok(())
            }
        }
        fn set_perm(&self, perm: FilePermissions) -> io::Result<()> {
            let attr = self.file_attr()?.attributes;
            if perm.readonly == (attr & FILE_ATTRIBUTE_READONLY != 0) {
                Ok(())
            } else if perm.readonly {
                self.set_attributes(attr | FILE_ATTRIBUTE_READONLY)
            } else {
                self.set_attributes(attr & !FILE_ATTRIBUTE_READONLY)
            }
        }

        fn handle(&self) -> &Handle {
            &self.handle
        }

        fn reparse_point<'a>(
            &self,
            space: &'a mut [u8; MAXIMUM_REPARSE_DATA_BUFFER_SIZE],
        ) -> io::Result<(DWORD, &'a REPARSE_DATA_BUFFER)> {
            unsafe {
                let mut bytes = 0;
                cvt({
                    DeviceIoControl(
                        self.handle.raw(),
                        FSCTL_GET_REPARSE_POINT,
                        ptr::null_mut(),
                        0,
                        space.as_mut_ptr() as *mut _,
                        space.len() as DWORD,
                        &mut bytes,
                        ptr::null_mut(),
                    )
                })?;
                Ok((bytes, &*(space.as_ptr() as *const REPARSE_DATA_BUFFER)))
            }
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    enum FileType {
        Dir,
        File,
        SymlinkFile,
        SymlinkDir,
        ReparsePoint,
        MountPoint,
    }

    impl FileType {
        fn new(attrs: DWORD, reparse_tag: DWORD) -> FileType {
            match (
                attrs & FILE_ATTRIBUTE_DIRECTORY != 0,
                attrs & FILE_ATTRIBUTE_REPARSE_POINT != 0,
                reparse_tag,
            ) {
                (false, false, _) => FileType::File,
                (true, false, _) => FileType::Dir,
                (false, true, IO_REPARSE_TAG_SYMLINK) => FileType::SymlinkFile,
                (true, true, IO_REPARSE_TAG_SYMLINK) => FileType::SymlinkDir,
                (true, true, IO_REPARSE_TAG_MOUNT_POINT) => FileType::MountPoint,
                (_, true, _) => FileType::ReparsePoint,
                // Note: if a _file_ has a reparse tag of the type IO_REPARSE_TAG_MOUNT_POINT it is
                // invalid, as junctions always have to be dirs. We set the filetype to ReparsePoint
                // to indicate it is something symlink-like, but not something you can follow.
            }
        }

        fn is_dir(&self) -> bool {
            *self == FileType::Dir
        }
        fn is_symlink_dir(&self) -> bool {
            *self == FileType::SymlinkDir || *self == FileType::MountPoint
        }
    }

    impl DirEntry {
        fn new(root: &Arc<PathBuf>, wfd: &WIN32_FIND_DATAW) -> Option<DirEntry> {
            let first_bytes = &wfd.cFileName[0..3];
            if first_bytes.starts_with(&[46, 0]) || first_bytes.starts_with(&[46, 46, 0]) {
                None
            } else {
                Some(DirEntry {
                    root: root.clone(),
                    data: *wfd,
                })
            }
        }

        fn path(&self) -> PathBuf {
            self.root.join(&self.file_name())
        }

        fn file_name(&self) -> OsString {
            let filename = truncate_utf16_at_nul(&self.data.cFileName);
            OsString::from_wide(filename)
        }

        fn file_type(&self) -> io::Result<FileType> {
            Ok(FileType::new(
                self.data.dwFileAttributes,
                /* reparse_tag = */ self.data.dwReserved0,
            ))
        }

        fn metadata(&self) -> io::Result<FileAttr> {
            Ok(FileAttr {
                attributes: self.data.dwFileAttributes,
                creation_time: self.data.ftCreationTime,
                last_access_time: self.data.ftLastAccessTime,
                last_write_time: self.data.ftLastWriteTime,
                file_size: ((self.data.nFileSizeHigh as u64) << 32)
                    | (self.data.nFileSizeLow as u64),
                reparse_tag: if self.data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                    // reserved unless this is a reparse point
                    self.data.dwReserved0
                } else {
                    0
                },
            })
        }
    }

    struct DirEntry {
        root: Arc<PathBuf>,
        data: WIN32_FIND_DATAW,
    }

    struct ReadDir {
        handle: FindNextFileHandle,
        root: Arc<PathBuf>,
        first: Option<WIN32_FIND_DATAW>,
    }

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
                    if FindNextFileW(self.handle.0, &mut wfd) == 0 {
                        if GetLastError() == ERROR_NO_MORE_FILES {
                            return None;
                        } else {
                            return Some(Err(io::Error::last_os_error()));
                        }
                    }
                    if let Some(e) = DirEntry::new(&self.root, &wfd) {
                        return Some(Ok(e));
                    }
                }
            }
        }
    }

    #[derive(Clone)]
    struct FileAttr {
        attributes: DWORD,
        creation_time: FILETIME,
        last_access_time: FILETIME,
        last_write_time: FILETIME,
        file_size: u64,
        reparse_tag: DWORD,
    }

    impl FileAttr {
        fn perm(&self) -> FilePermissions {
            FilePermissions {
                readonly: self.attributes & FILE_ATTRIBUTE_READONLY != 0,
            }
        }

        fn file_type(&self) -> FileType {
            FileType::new(self.attributes, self.reparse_tag)
        }

        fn is_reparse_point(&self) -> bool {
            self.attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0
        }
    }

    #[repr(C)]
    struct REPARSE_DATA_BUFFER {
        ReparseTag: c_uint,
        ReparseDataLength: c_ushort,
        Reserved: c_ushort,
        rest: (),
    }

    const MAXIMUM_REPARSE_DATA_BUFFER_SIZE: usize = 16 * 1024;

    /// An owned container for `HANDLE` object, closing them on Drop.
    ///
    /// All methods are inherited through a `Deref` impl to `RawHandle`
    struct Handle(RawHandle);

    use std::ops::Deref;

    /// A wrapper type for `HANDLE` objects to give them proper Send/Sync inference
    /// as well as Rust-y methods.
    ///
    /// This does **not** drop the handle when it goes out of scope, use `Handle`
    /// instead for that.
    #[derive(Copy, Clone)]
    struct RawHandle(HANDLE);

    unsafe impl Send for RawHandle {}
    unsafe impl Sync for RawHandle {}

    impl Handle {
        fn new(handle: HANDLE) -> Handle {
            Handle(RawHandle::new(handle))
        }
    }

    impl Deref for Handle {
        type Target = RawHandle;
        fn deref(&self) -> &RawHandle {
            &self.0
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.raw());
            }
        }
    }

    impl RawHandle {
        fn new(handle: HANDLE) -> RawHandle {
            RawHandle(handle)
        }

        fn raw(&self) -> HANDLE {
            self.0
        }
    }

    struct FindNextFileHandle(HANDLE);

    fn get_path(f: &File) -> io::Result<PathBuf> {
        fill_utf16_buf(
            |buf, sz| unsafe {
                GetFinalPathNameByHandleW(f.handle.raw(), buf, sz, VOLUME_NAME_DOS)
            },
            |buf| PathBuf::from(OsString::from_wide(buf)),
        )
    }

    fn move_item(file: &File, ctx: &mut RmdirContext) -> io::Result<()> {
        let mut tmpname = ctx.base_dir.join(format! {"rm-{}", ctx.counter});
        ctx.counter += 1;
        // Try to rename the file. If it already exists, just retry with an other
        // filename.
        while let Err(err) = file.rename(tmpname.as_ref(), false) {
            if err.kind() != io::ErrorKind::AlreadyExists {
                return Err(err);
            };
            tmpname = ctx.base_dir.join(format!("rm-{}", ctx.counter));
            ctx.counter += 1;
        }
        Ok(())
    }

    fn set_perm(path: &Path, perm: FilePermissions) -> io::Result<()> {
        let mut opts = OpenOptions::new();
        opts.access_mode(FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES);
        opts.custom_flags(FILE_FLAG_BACKUP_SEMANTICS);
        let file = File::open(path, &opts)?;
        file.set_perm(perm)
    }

    const VOLUME_NAME_DOS: DWORD = 0x0;
}
