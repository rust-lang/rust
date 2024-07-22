use crate::ffi::{CString, OsString};
use crate::fmt;
use crate::hash::Hash;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sys::time::SystemTime;
use crate::sys::unsupported;

#[derive(Debug)]
struct FileDesc(*mut vex_sdk::FIL);

pub struct File {
    fd: FileDesc,
}

//TODO: We may be able to get some of this info
#[derive(Clone)]
pub struct FileAttr {
    size: u64,
}

pub struct ReadDir(!);

pub struct DirEntry(!);

#[derive(Clone, Debug)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create_new: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilePermissions;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct FileType {
    is_dir: bool,
}

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    /// Creates a FileAttr by getting data from an opened file.
    fn from_fd(fd: *mut vex_sdk::FIL) -> io::Result<Self> {
        let size = unsafe { vex_sdk::vexFileSize(fd) };

        if size >= 0 {
            Ok(Self { size: size as u64 })
        } else {
            Err(io::Error::new(io::ErrorKind::NotSeekable, "Failed to seek file"))
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions
    }

    pub fn file_type(&self) -> FileType {
        FileType { is_dir: false }
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        unsupported()
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        unsupported()
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        unsupported()
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        false
    }

    pub fn set_readonly(&mut self, _readonly: bool) {
        panic!("Perimissions do not exist")
    }
}

impl FileTimes {
    pub fn set_accessed(&mut self, _t: SystemTime) {}
    pub fn set_modified(&mut self, _t: SystemTime) {}
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.is_dir
    }

    pub fn is_file(&self) -> bool {
        !self.is_dir
    }

    pub fn is_symlink(&self) -> bool {
        false
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        self.0
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.0
    }

    pub fn file_name(&self) -> OsString {
        self.0
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        self.0
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        self.0
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions { read: false, write: false, append: false, truncate: false, create_new: false }
    }

    pub fn read(&mut self, read: bool) {
        self.read = read;
    }
    pub fn write(&mut self, write: bool) {
        self.write = write;
    }
    pub fn append(&mut self, append: bool) {
        self.append = append;
    }
    pub fn truncate(&mut self, truncate: bool) {
        self.truncate = truncate;
    }
    pub fn create(&mut self, create: bool) {
        self.write = create;
    }
    pub fn create_new(&mut self, create_new: bool) {
        self.create_new = create_new;
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        // Mount sdcard volume as FAT filesystem
        map_fresult(unsafe { vex_sdk::vexFileMountSD() })?;

        let path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Path contained a null byte")
        })?;

        if opts.write && opts.read {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Files cannot be opened with read and write access",
            ));
        }
        if opts.create_new {
            let file_exists = unsafe { vex_sdk::vexFileStatus(path.as_ptr()) };
            if file_exists != 0 {
                return Err(io::Error::new(io::ErrorKind::AlreadyExists, "File already exists"));
            }
        }

        let file = if opts.read && !opts.write {
            // The second argument to this function is ignored.
            // Open in read only mode
            unsafe { vex_sdk::vexFileOpen(path.as_ptr(), c"".as_ptr()) }
        } else if opts.write && opts.append {
            // Open in read/write and append mode
            unsafe { vex_sdk::vexFileOpenWrite(path.as_ptr()) }
        } else if opts.write && opts.truncate {
            // Open in read/write mode
            unsafe { vex_sdk::vexFileOpenCreate(path.as_ptr()) }
        } else if opts.write {
            // Open in read/write and overwrite mode
            unsafe {
                // Open in read/write and append mode
                let fd = vex_sdk::vexFileOpenWrite(path.as_ptr());
                // Seek to beginning of the file
                vex_sdk::vexFileSeek(fd, 0, 0);

                fd
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Files cannot be opened without read or write access",
            ));
        };

        if file.is_null() {
            Err(io::Error::new(io::ErrorKind::NotFound, "Could not open file"))
        } else {
            Ok(Self { fd: FileDesc(file) })
        }
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        FileAttr::from_fd(self.fd.0)
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.flush()
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.flush()
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        unsupported()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len() as _;
        let buf_ptr = buf.as_mut_ptr();
        let read = unsafe { vex_sdk::vexFileRead(buf_ptr.cast(), 1, len, self.fd.0) };
        if read < 0 {
            Err(io::Error::new(io::ErrorKind::Other, "Could not read from file"))
        } else {
            Ok(read as usize)
        }
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|b| self.read(b), bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|b| self.read(b), cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len();
        let buf_ptr = buf.as_ptr();
        let written =
            unsafe { vex_sdk::vexFileWrite(buf_ptr.cast_mut().cast(), 1, len as _, self.fd.0) };
        if written < 0 {
            Err(io::Error::new(io::ErrorKind::Other, "Could not write to file"))
        } else {
            Ok(written as usize)
        }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|b| self.write(b), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        unsafe {
            vex_sdk::vexFileSync(self.fd.0);
        }
        Ok(())
    }

    fn tell(&self) -> io::Result<u64> {
        let position = unsafe { vex_sdk::vexFileTell(self.fd.0) };
        position.try_into().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Failed to get current location in file")
        })
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        const SEEK_SET: i32 = 0;
        const SEEK_CUR: i32 = 1;
        const SEEK_END: i32 = 2;

        fn try_convert_offset<T: TryInto<u32>>(offset: T) -> io::Result<u32> {
            offset.try_into().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Cannot seek to an offset too large to fit in a 32 bit integer",
                )
            })
        }

        match pos {
            SeekFrom::Start(offset) => unsafe {
                map_fresult(vex_sdk::vexFileSeek(self.fd.0, try_convert_offset(offset)?, SEEK_SET))?
            },

            // The VEX SDK does not allow seeking with negative offsets.
            // That means we need to calculate the offset from the start for both of these.
            SeekFrom::End(offset) => unsafe {
                // If our offset is positive, everything is easy
                if offset >= 0 {
                    map_fresult(vex_sdk::vexFileSeek(
                        self.fd.0,
                        try_convert_offset(offset)?,
                        SEEK_END,
                    ))?
                } else {
                    // Get the position of the end of the file...
                    map_fresult(vex_sdk::vexFileSeek(
                        self.fd.0,
                        try_convert_offset(offset)?,
                        SEEK_END,
                    ))?;
                    // The number returned by the VEX SDK tell is stored as a 32 bit interger,
                    // and therefore this conversion cannot fail.
                    let position = self.tell()? as i64;

                    // Offset from that position
                    let new_position = position + offset;
                    map_fresult(vex_sdk::vexFileSeek(
                        self.fd.0,
                        try_convert_offset(new_position)?,
                        SEEK_SET,
                    ))?
                }
            },
            SeekFrom::Current(offset) => unsafe {
                if offset >= 0 {
                    map_fresult(vex_sdk::vexFileSeek(
                        self.fd.0,
                        try_convert_offset(offset)?,
                        SEEK_CUR,
                    ))?
                } else {
                    let position = self.tell()? as i64;

                    let new_position = position + offset;
                    map_fresult(vex_sdk::vexFileSeek(
                        self.fd.0,
                        try_convert_offset(new_position)?,
                        SEEK_SET,
                    ))?
                }
            },
        }

        Ok(self.tell()?)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        unsupported()
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        unsupported()
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        unsupported()
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, _p: &Path) -> io::Result<()> {
        unsupported()
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File").field("fd", &self.fd.0).finish()
    }
}
impl Drop for File {
    fn drop(&mut self) {
        unsafe { vex_sdk::vexFileClose(self.fd.0) };
    }
}

pub fn readdir(_p: &Path) -> io::Result<ReadDir> {
    todo!()
}

pub fn unlink(_p: &Path) -> io::Result<()> {
    unsupported()
}

pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> {
    unsupported()
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    unsupported()
}

pub fn rmdir(_p: &Path) -> io::Result<()> {
    unsupported()
}

pub fn remove_dir_all(_path: &Path) -> io::Result<()> {
    unsupported()
}

pub fn try_exists(path: &Path) -> io::Result<bool> {
    let path = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Path contained a null byte"))?;

    let file_exists = unsafe { vex_sdk::vexFileStatus(path.as_ptr()) };
    if file_exists != 0 { Ok(true) } else { Ok(false) }
}

pub fn readlink(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn symlink(_original: &Path, _link: &Path) -> io::Result<()> {
    unsupported()
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    unsupported()
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    let file = File::open(p, &opts)?;
    let fd = file.fd.0;

    FileAttr::from_fd(fd)
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    // Symlinks aren't supported in our filesystem
    stat(p)
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    use crate::fs::File;

    let mut reader = File::open(from)?;
    let mut writer = File::create(to)?;

    io::copy(&mut reader, &mut writer)
}

fn map_fresult(fresult: vex_sdk::FRESULT) -> io::Result<()> {
    // VEX presumably uses a derivative of FatFs (most likely the xilffs library)
    // for sdcard filesystem functions.
    //
    // Documentation for each FRESULT originates from here:
    // <http://elm-chan.org/fsw/ff/doc/rc.html>
    match fresult {
        vex_sdk::FRESULT::FR_OK => Ok(()),
        vex_sdk::FRESULT::FR_DISK_ERR => Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            "internal function reported an unrecoverable hard error",
        )),
        vex_sdk::FRESULT::FR_INT_ERR => Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            "assertion failed and an insanity is detected in the internal process",
        )),
        vex_sdk::FRESULT::FR_NOT_READY => Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            "the storage device could not be prepared to work",
        )),
        vex_sdk::FRESULT::FR_NO_FILE => {
            Err(io::Error::new(io::ErrorKind::NotFound, "could not find the file in the directory"))
        }
        vex_sdk::FRESULT::FR_NO_PATH => Err(io::Error::new(
            io::ErrorKind::NotFound,
            "a directory in the path name could not be found",
        )),
        vex_sdk::FRESULT::FR_INVALID_NAME => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the given string is invalid as a path name",
        )),
        vex_sdk::FRESULT::FR_DENIED => Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "the required access for this operation was denied",
        )),
        vex_sdk::FRESULT::FR_EXIST => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "an object with the same name already exists in the directory",
        )),
        vex_sdk::FRESULT::FR_INVALID_OBJECT => Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            "invalid or null file/directory object",
        )),
        vex_sdk::FRESULT::FR_WRITE_PROTECTED => Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "a write operation was performed on write-protected media",
        )),
        vex_sdk::FRESULT::FR_INVALID_DRIVE => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "an invalid drive number was specified in the path name",
        )),
        vex_sdk::FRESULT::FR_NOT_ENABLED => Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            "work area for the logical drive has not been registered",
        )),
        vex_sdk::FRESULT::FR_NO_FILESYSTEM => Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            "valid FAT volume could not be found on the drive",
        )),
        vex_sdk::FRESULT::FR_MKFS_ABORTED => {
            Err(io::Error::new(io::ErrorKind::Uncategorized, "failed to create filesystem volume"))
        }
        vex_sdk::FRESULT::FR_TIMEOUT => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "the function was canceled due to a timeout of thread-safe control",
        )),
        vex_sdk::FRESULT::FR_LOCKED => Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            "the operation to the object was rejected by file sharing control",
        )),
        vex_sdk::FRESULT::FR_NOT_ENOUGH_CORE => {
            Err(io::Error::new(io::ErrorKind::OutOfMemory, "not enough memory for the operation"))
        }
        vex_sdk::FRESULT::FR_TOO_MANY_OPEN_FILES => Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            "maximum number of open files has been reached",
        )),
        vex_sdk::FRESULT::FR_INVALID_PARAMETER => {
            Err(io::Error::new(io::ErrorKind::InvalidInput, "a given parameter was invalid"))
        }
        _ => unreachable!(), // C-style enum
    }
}
