// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Blocking Windows-based file I/O

use alloc::arc::Arc;
use libc::{mod, c_int};

use c_str::CString;
use mem;
use sys::os::fill_utf16_buf_and_decode;
use path;
use ptr;
use str;
use io;

use prelude::*;
use sys;
use sys::os;
use sys_common::{keep_going, eof, mkerr_libc};

use io::{FilePermission, Write, UnstableFileStat, Open, FileAccess, FileMode};
use io::{IoResult, IoError, FileStat, SeekStyle};
use io::{Read, Truncate, SeekCur, SeekSet, ReadWrite, SeekEnd, Append};

pub type fd_t = libc::c_int;

pub struct FileDesc {
    /// The underlying C file descriptor.
    pub fd: fd_t,

    /// Whether to close the file descriptor on drop.
    close_on_drop: bool,
}

impl FileDesc {
    pub fn new(fd: fd_t, close_on_drop: bool) -> FileDesc {
        FileDesc { fd: fd, close_on_drop: close_on_drop }
    }

    pub fn read(&self, buf: &mut [u8]) -> IoResult<uint> {
        let mut read = 0;
        let ret = unsafe {
            libc::ReadFile(self.handle(), buf.as_ptr() as libc::LPVOID,
                           buf.len() as libc::DWORD, &mut read,
                           ptr::null_mut())
        };
        if ret != 0 {
            Ok(read as uint)
        } else {
            Err(super::last_error())
        }
    }

    pub fn write(&self, buf: &[u8]) -> IoResult<()> {
        let mut cur = buf.as_ptr();
        let mut remaining = buf.len();
        while remaining > 0 {
            let mut amt = 0;
            let ret = unsafe {
                libc::WriteFile(self.handle(), cur as libc::LPVOID,
                                remaining as libc::DWORD, &mut amt,
                                ptr::null_mut())
            };
            if ret != 0 {
                remaining -= amt as uint;
                cur = unsafe { cur.offset(amt as int) };
            } else {
                return Err(super::last_error())
            }
        }
        Ok(())
    }

    pub fn fd(&self) -> fd_t { self.fd }

    pub fn handle(&self) -> libc::HANDLE {
        unsafe { libc::get_osfhandle(self.fd()) as libc::HANDLE }
    }

    // A version of seek that takes &self so that tell can call it
    //   - the private seek should of course take &mut self.
    fn seek_common(&self, pos: i64, style: SeekStyle) -> IoResult<u64> {
        let whence = match style {
            SeekSet => libc::FILE_BEGIN,
            SeekEnd => libc::FILE_END,
            SeekCur => libc::FILE_CURRENT,
        };
        unsafe {
            let mut newpos = 0;
            match libc::SetFilePointerEx(self.handle(), pos, &mut newpos, whence) {
                0 => Err(super::last_error()),
                _ => Ok(newpos as u64),
            }
        }
    }

    pub fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<u64> {
        self.seek_common(pos, style)
    }

    pub fn tell(&self) -> IoResult<u64> {
        self.seek_common(0, SeekCur)
    }

    pub fn fsync(&mut self) -> IoResult<()> {
        super::mkerr_winbool(unsafe {
            libc::FlushFileBuffers(self.handle())
        })
    }

    pub fn datasync(&mut self) -> IoResult<()> { return self.fsync(); }

    pub fn truncate(&mut self, offset: i64) -> IoResult<()> {
        let orig_pos = try!(self.tell());
        let _ = try!(self.seek(offset, SeekSet));
        let ret = unsafe {
            match libc::SetEndOfFile(self.handle()) {
                0 => Err(super::last_error()),
                _ => Ok(())
            }
        };
        let _ = self.seek(orig_pos as i64, SeekSet);
        return ret;
    }

    pub fn fstat(&self) -> IoResult<io::FileStat> {
        let mut stat: libc::stat = unsafe { mem::zeroed() };
        match unsafe { libc::fstat(self.fd(), &mut stat) } {
            0 => Ok(mkstat(&stat)),
            _ => Err(super::last_error()),
        }
    }

    /// Extract the actual filedescriptor without closing it.
    pub fn unwrap(self) -> fd_t {
        let fd = self.fd;
        unsafe { mem::forget(self) };
        fd
    }
}

impl Drop for FileDesc {
    fn drop(&mut self) {
        // closing stdio file handles makes no sense, so never do it. Also, note
        // that errors are ignored when closing a file descriptor. The reason
        // for this is that if an error occurs we don't actually know if the
        // file descriptor was closed or not, and if we retried (for something
        // like EINTR), we might close another valid file descriptor (opened
        // after we closed ours.
        if self.close_on_drop && self.fd > libc::STDERR_FILENO {
            let n = unsafe { libc::close(self.fd) };
            if n != 0 {
                println!("error {} when closing file descriptor {}", n, self.fd);
            }
        }
    }
}

pub fn to_utf16(s: &Path) -> IoResult<Vec<u16>> {
    sys::to_utf16(s.as_str())
}

pub fn open(path: &Path, fm: FileMode, fa: FileAccess) -> IoResult<FileDesc> {
    // Flags passed to open_osfhandle
    let flags = match fm {
        Open => 0,
        Append => libc::O_APPEND,
        Truncate => libc::O_TRUNC,
    };
    let flags = match fa {
        Read => flags | libc::O_RDONLY,
        Write => flags | libc::O_WRONLY | libc::O_CREAT,
        ReadWrite => flags | libc::O_RDWR | libc::O_CREAT,
    };
    let mut dwDesiredAccess = match fa {
        Read => libc::FILE_GENERIC_READ,
        Write => libc::FILE_GENERIC_WRITE,
        ReadWrite => libc::FILE_GENERIC_READ | libc::FILE_GENERIC_WRITE
    };

    // libuv has a good comment about this, but the basic idea is what we try to
    // emulate unix semantics by enabling all sharing by allowing things such as
    // deleting a file while it's still open.
    let dwShareMode = libc::FILE_SHARE_READ | libc::FILE_SHARE_WRITE |
                      libc::FILE_SHARE_DELETE;

    let dwCreationDisposition = match (fm, fa) {
        (Truncate, Read) => libc::TRUNCATE_EXISTING,
        (Truncate, _) => libc::CREATE_ALWAYS,
        (Open, Read) => libc::OPEN_EXISTING,
        (Open, _) => libc::OPEN_ALWAYS,
        (Append, Read) => {
            dwDesiredAccess |= libc::FILE_APPEND_DATA;
            libc::OPEN_EXISTING
        }
        (Append, _) => {
            dwDesiredAccess &= !libc::FILE_WRITE_DATA;
            dwDesiredAccess |= libc::FILE_APPEND_DATA;
            libc::OPEN_ALWAYS
        }
    };

    let mut dwFlagsAndAttributes = libc::FILE_ATTRIBUTE_NORMAL;
    // Compat with unix, this allows opening directories (see libuv)
    dwFlagsAndAttributes |= libc::FILE_FLAG_BACKUP_SEMANTICS;

    let path = try!(to_utf16(path));
    let handle = unsafe {
        libc::CreateFileW(path.as_ptr(),
                          dwDesiredAccess,
                          dwShareMode,
                          ptr::null_mut(),
                          dwCreationDisposition,
                          dwFlagsAndAttributes,
                          ptr::null_mut())
    };
    if handle == libc::INVALID_HANDLE_VALUE {
        Err(super::last_error())
    } else {
        let fd = unsafe {
            libc::open_osfhandle(handle as libc::intptr_t, flags)
        };
        if fd < 0 {
            let _ = unsafe { libc::CloseHandle(handle) };
            Err(super::last_error())
        } else {
            Ok(FileDesc::new(fd, true))
        }
    }
}

pub fn mkdir(p: &Path, _mode: uint) -> IoResult<()> {
    let p = try!(to_utf16(p));
    super::mkerr_winbool(unsafe {
        // FIXME: turn mode into something useful? #2623
        libc::CreateDirectoryW(p.as_ptr(), ptr::null_mut())
    })
}

pub fn readdir(p: &Path) -> IoResult<Vec<Path>> {
    fn prune(root: &Path, dirs: Vec<Path>) -> Vec<Path> {
        dirs.into_iter().filter(|path| {
            path.as_vec() != b"." && path.as_vec() != b".."
        }).map(|path| root.join(path)).collect()
    }

    let star = p.join("*");
    let path = try!(to_utf16(&star));

    unsafe {
        let mut wfd = mem::zeroed();
        let find_handle = libc::FindFirstFileW(path.as_ptr(), &mut wfd);
        if find_handle != libc::INVALID_HANDLE_VALUE {
            let mut paths = vec![];
            let mut more_files = 1 as libc::BOOL;
            while more_files != 0 {
                {
                    let filename = os::truncate_utf16_at_nul(&wfd.cFileName);
                    match String::from_utf16(filename) {
                        Some(filename) => paths.push(Path::new(filename)),
                        None => {
                            assert!(libc::FindClose(find_handle) != 0);
                            return Err(IoError {
                                kind: io::InvalidInput,
                                desc: "path was not valid UTF-16",
                                detail: Some(format!("path was not valid UTF-16: {}", filename)),
                            })
                        }, // FIXME #12056: Convert the UCS-2 to invalid utf-8 instead of erroring
                    }
                }
                more_files = libc::FindNextFileW(find_handle, &mut wfd);
            }
            assert!(libc::FindClose(find_handle) != 0);
            Ok(prune(p, paths))
        } else {
            Err(super::last_error())
        }
    }
}

pub fn unlink(p: &Path) -> IoResult<()> {
    fn do_unlink(p_utf16: &Vec<u16>) -> IoResult<()> {
        super::mkerr_winbool(unsafe { libc::DeleteFileW(p_utf16.as_ptr()) })
    }

    let p_utf16 = try!(to_utf16(p));
    let res = do_unlink(&p_utf16);
    match res {
        Ok(()) => Ok(()),
        Err(e) => {
            // FIXME: change the code below to use more direct calls
            // than `stat` and `chmod`, to avoid re-conversion to
            // utf16 etc.

            // On unix, a readonly file can be successfully removed. On windows,
            // however, it cannot. To keep the two platforms in line with
            // respect to their behavior, catch this case on windows, attempt to
            // change it to read-write, and then remove the file.
            if e.kind == io::PermissionDenied {
                let stat = match stat(p) {
                    Ok(stat) => stat,
                    Err(..) => return Err(e),
                };
                if stat.perm.intersects(io::USER_WRITE) { return Err(e) }

                match chmod(p, (stat.perm | io::USER_WRITE).bits() as uint) {
                    Ok(()) => do_unlink(&p_utf16),
                    Err(..) => {
                        // Try to put it back as we found it
                        let _ = chmod(p, stat.perm.bits() as uint);
                        Err(e)
                    }
                }
            } else {
                Err(e)
            }
        }
    }
}

pub fn rename(old: &Path, new: &Path) -> IoResult<()> {
    let old = try!(to_utf16(old));
    let new = try!(to_utf16(new));
    super::mkerr_winbool(unsafe {
        libc::MoveFileExW(old.as_ptr(), new.as_ptr(), libc::MOVEFILE_REPLACE_EXISTING)
    })
}

pub fn chmod(p: &Path, mode: uint) -> IoResult<()> {
    let p = try!(to_utf16(p));
    mkerr_libc(unsafe {
        libc::wchmod(p.as_ptr(), mode as libc::c_int)
    })
}

pub fn rmdir(p: &Path) -> IoResult<()> {
    let p = try!(to_utf16(p));
    mkerr_libc(unsafe { libc::wrmdir(p.as_ptr()) })
}

pub fn chown(_p: &Path, _uid: int, _gid: int) -> IoResult<()> {
    // libuv has this as a no-op, so seems like this should as well?
    Ok(())
}

pub fn readlink(p: &Path) -> IoResult<Path> {
    // FIXME: I have a feeling that this reads intermediate symlinks as well.
    use sys::c::compat::kernel32::GetFinalPathNameByHandleW;
    let p = try!(to_utf16(p));
    let handle = unsafe {
        libc::CreateFileW(p.as_ptr(),
                          libc::GENERIC_READ,
                          libc::FILE_SHARE_READ,
                          ptr::null_mut(),
                          libc::OPEN_EXISTING,
                          libc::FILE_ATTRIBUTE_NORMAL,
                          ptr::null_mut())
    };
    if handle == libc::INVALID_HANDLE_VALUE {
        return Err(super::last_error())
    }
    // Specify (sz - 1) because the documentation states that it's the size
    // without the null pointer
    let ret = fill_utf16_buf_and_decode(|buf, sz| unsafe {
        GetFinalPathNameByHandleW(handle,
                                  buf as *const u16,
                                  sz - 1,
                                  libc::VOLUME_NAME_DOS)
    });
    let ret = match ret {
        Some(ref s) if s.starts_with(r"\\?\") => { // "
            Ok(Path::new(s.slice_from(4)))
        }
        Some(s) => Ok(Path::new(s)),
        None => Err(super::last_error()),
    };
    assert!(unsafe { libc::CloseHandle(handle) } != 0);
    return ret;
}

pub fn symlink(src: &Path, dst: &Path) -> IoResult<()> {
    use sys::c::compat::kernel32::CreateSymbolicLinkW;
    let src = try!(to_utf16(src));
    let dst = try!(to_utf16(dst));
    super::mkerr_winbool(unsafe {
        CreateSymbolicLinkW(dst.as_ptr(), src.as_ptr(), 0) as libc::BOOL
    })
}

pub fn link(src: &Path, dst: &Path) -> IoResult<()> {
    let src = try!(to_utf16(src));
    let dst = try!(to_utf16(dst));
    super::mkerr_winbool(unsafe {
        libc::CreateHardLinkW(dst.as_ptr(), src.as_ptr(), ptr::null_mut())
    })
}

fn mkstat(stat: &libc::stat) -> FileStat {
    FileStat {
        size: stat.st_size as u64,
        kind: match (stat.st_mode as libc::c_int) & libc::S_IFMT {
            libc::S_IFREG => io::FileType::RegularFile,
            libc::S_IFDIR => io::FileType::Directory,
            libc::S_IFIFO => io::FileType::NamedPipe,
            libc::S_IFBLK => io::FileType::BlockSpecial,
            libc::S_IFLNK => io::FileType::Symlink,
            _ => io::FileType::Unknown,
        },
        perm: FilePermission::from_bits_truncate(stat.st_mode as u32),
        created: stat.st_ctime as u64,
        modified: stat.st_mtime as u64,
        accessed: stat.st_atime as u64,
        unstable: UnstableFileStat {
            device: stat.st_dev as u64,
            inode: stat.st_ino as u64,
            rdev: stat.st_rdev as u64,
            nlink: stat.st_nlink as u64,
            uid: stat.st_uid as u64,
            gid: stat.st_gid as u64,
            blksize:0,
            blocks: 0,
            flags: 0,
            gen: 0,
        },
    }
}

pub fn stat(p: &Path) -> IoResult<FileStat> {
    let mut stat: libc::stat = unsafe { mem::zeroed() };
    let p = try!(to_utf16(p));
    match unsafe { libc::wstat(p.as_ptr(), &mut stat) } {
        0 => Ok(mkstat(&stat)),
        _ => Err(super::last_error()),
    }
}

// FIXME: move this to platform-specific modules (for now)?
pub fn lstat(_p: &Path) -> IoResult<FileStat> {
    // FIXME: implementation is missing
    Err(super::unimpl())
}

pub fn utime(p: &Path, atime: u64, mtime: u64) -> IoResult<()> {
    let mut buf = libc::utimbuf {
        actime: atime as libc::time64_t,
        modtime: mtime as libc::time64_t,
    };
    let p = try!(to_utf16(p));
    mkerr_libc(unsafe {
        libc::wutime(p.as_ptr(), &mut buf)
    })
}
