
use io::{self, Error, ErrorKind};
use libc;
use mem;
use path::Path;
use ptr;
use sys::{cvt, cvt_r};
use cmp;
use fs::File;
use sync::atomic::{AtomicBool, Ordering};
use super::ext::fs::MetadataExt;
use super::ext::io::AsRawFd;



// Kernel prior to 4.5 don't have copy_file_range
// We store the availability in a global to avoid unnecessary syscalls
static HAS_COPY_FILE_RANGE: AtomicBool = AtomicBool::new(true);

unsafe fn copy_file_range(
    fd_in: libc::c_int,
    off_in: *mut libc::loff_t,
    fd_out: libc::c_int,
    off_out: *mut libc::loff_t,
    len: libc::size_t,
    flags: libc::c_uint,
) -> libc::c_long {
    libc::syscall(
        libc::SYS_copy_file_range,
        fd_in,
        off_in,
        fd_out,
        off_out,
        len,
        flags,
    )
}

/// Corresponds to lseek(2) `wence`. This exists in std, but doesn't support sparse-files.
#[allow(dead_code)]
pub enum Wence {
    Set = libc::SEEK_SET as isize,
    Cur = libc::SEEK_CUR as isize,
    End = libc::SEEK_END as isize,
    Data = libc::SEEK_DATA as isize,
    Hole = libc::SEEK_HOLE as isize,
}

#[derive(PartialEq, Debug)]
pub enum SeekOff {
    Offset(u64),
    EOF
}

pub fn lseek(fd: &File, off: i64, wence: Wence) -> io::Result<SeekOff> {
    let r = unsafe {
        libc::lseek64(
            fd.as_raw_fd(),
            off,
            wence as libc::c_int
        )
    };

    if r == -1 {
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(errno) if errno == libc::ENXIO => {
                Ok(SeekOff::EOF)
            }
            _ => Err(err.into())
        }

    } else {
        Ok(SeekOff::Offset(r as u64))
    }

}

pub fn allocate_file(fd: &File, len: u64) -> io::Result<()> {
    cvt_r(|| unsafe {libc::ftruncate64(fd.as_raw_fd(), len as i64)})?;
    Ok(())
}

/// Version of copy_file_range that defers offset-management to the
/// syscall. see copy_file_range(2) for details.
pub fn copy_file_bytes(infd: &File, outfd: &File, bytes: u64) -> io::Result<u64> {
    let r = cvt(unsafe {
        copy_file_range(
            infd.as_raw_fd(),
            ptr::null_mut(),
            outfd.as_raw_fd(),
            ptr::null_mut(),
            bytes as usize,
            0,
        ) as i64
    })?;
    Ok(r as u64)
}

/// Copy len bytes from whereever the descriptor cursors are set.
fn copy_range(infd: &File, outfd: &File, len: u64) -> io::Result<u64> {
    let mut written = 0u64;
    while written < len {
        let result = copy_file_bytes(&infd, &outfd, len - written)?;
        written += result;
    }
    Ok(written)
}

fn next_sparse_segments(fd: &File, pos: u64) -> io::Result<(u64, u64)> {
    let next_data = match lseek(fd, pos as i64, Wence::Data)? {
        SeekOff::Offset(off) => off,
        SeekOff::EOF => fd.metadata()?.len()
    };
    let next_hole = match lseek(fd, next_data as i64, Wence::Hole)? {
        SeekOff::Offset(off) => off,
        SeekOff::EOF => fd.metadata()?.len()
    };

    Ok((next_data, next_hole))
}

fn copy_sparse(infd: &File, outfd: &File) -> io::Result<u64> {
    let len = infd.metadata()?.len();
    allocate_file(&outfd, len)?;

    let mut pos = 0;

    while pos < len {
        let (next_data, next_hole) = next_sparse_segments(infd, pos)?;
        lseek(infd, next_data as i64, Wence::Set)?;
        lseek(outfd, next_data as i64, Wence::Set)?;

        let _written = copy_range(infd, outfd, next_hole - next_data)?;
        pos = next_hole;
    }

    Ok(len)
}


// Version of copy_file_range(2) that copies the give range to the
// same place in the target file. If off is None then use nul to
// tell copy_file_range() track the file offset. See the manpage
// for details.
fn copy_file_chunk(reader: &File, writer: &File, off: Option<i64>, bytes_to_copy: usize) -> io::Result<libc::c_long> {
    let mut off_val = off.unwrap_or(0);
    let copy_result = unsafe {
        let off_ptr = if off.is_some() {
            &mut off_val as *mut i64
        } else {
            ptr::null_mut()
        };
        cvt(copy_file_range(reader.as_raw_fd(),
                            off_ptr,
                            writer.as_raw_fd(),
                            off_ptr,
                            bytes_to_copy,
                            0)
        )
    };
    if let Err(ref copy_err) = copy_result {
        match copy_err.raw_os_error() {
            Some(libc::ENOSYS) | Some(libc::EPERM) => {
                HAS_COPY_FILE_RANGE.store(false, Ordering::Relaxed);
            }
            _ => {}
        }
    }
    copy_result
}

fn is_sparse(fd: &File) -> io::Result<bool> {
    let mut stat: libc::stat = unsafe { mem::uninitialized() };
    cvt(unsafe { libc::fstat(fd.as_raw_fd(), &mut stat) })?;
    Ok(stat.st_blocks < stat.st_size / stat.st_blksize)
}


pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {

    if !from.is_file() {
        return Err(Error::new(ErrorKind::InvalidInput,
                              "the source path is not an existing regular file"))
    }

    let mut reader = File::open(from)?;
    let mut writer = File::create(to)?;
    let (perm, len) = {
        let metadata = reader.metadata()?;
        (metadata.permissions(), metadata.size())
    };
    let _sparse = is_sparse(&reader)?;

    let has_copy_file_range = HAS_COPY_FILE_RANGE.load(Ordering::Relaxed);
    let mut written = 0u64;
    while written < len {
        let copy_result = if has_copy_file_range {
            let bytes_to_copy = cmp::min(len - written, usize::max_value() as u64) as usize;
            copy_file_chunk(&reader, &writer, None, bytes_to_copy)

        } else {
            Err(io::Error::from_raw_os_error(libc::ENOSYS))
        };
        match copy_result {
            Ok(ret) => written += ret as u64,
            Err(err) => {
                match err.raw_os_error() {
                    Some(os_err) if os_err == libc::ENOSYS
                                 || os_err == libc::EXDEV
                                 || os_err == libc::EPERM => {
                        // Try fallback io::copy if either:
                        // - Kernel version is < 4.5 (ENOSYS)
                        // - Files are mounted on different fs (EXDEV)
                        // - copy_file_range is disallowed, for example by seccomp (EPERM)
                        assert_eq!(written, 0);
                        let ret = io::copy(&mut reader, &mut writer)?;
                        writer.set_permissions(perm)?;
                        return Ok(ret)
                    },
                    _ => return Err(err),
                }
            }
        }
    }
    writer.set_permissions(perm)?;
    Ok(written)
}


#[cfg(test)]
mod tests {
    use super::*;
    extern crate tempfile;
    use self::tempfile::{tempdir, TempDir};
    use fs::{read, OpenOptions};
    use io::{Seek, SeekFrom, Write};

    fn create_sparse(file: &String) {
        let fd = File::create(file).unwrap();
        cvt(unsafe {libc::ftruncate64(fd.as_raw_fd(), 1024*1024)}).unwrap();
    }

    fn create_sparse_with_data(file: &String, head: u64, tail: u64) -> u64 {
        let data = "c00lc0d3";
        let len = 4096u64 * 4096 + data.len() as u64 + tail;

        {
            let fd = File::create(file).unwrap();
            cvt(unsafe {libc::ftruncate64(fd.as_raw_fd(), len as i64)}).unwrap();
        }

        let mut fd = OpenOptions::new()
            .write(true)
            .append(false)
            .open(&file).unwrap();

        fd.seek(SeekFrom::Start(head)).unwrap();
        write!(fd, "{}", data);

        fd.seek(SeekFrom::Start(1024*4096)).unwrap();
        write!(fd, "{}", data);

        fd.seek(SeekFrom::Start(4096*4096)).unwrap();
        write!(fd, "{}", data);

        len
    }


    fn tmps(dir: &TempDir) -> (String, String) {
        let file = dir.path().join("sparse.bin");
        let from = dir.path().join("from.txt");
        (file.to_str().unwrap().to_string(),
         from.to_str().unwrap().to_string())
    }


    #[test]
    fn test_sparse_detection() {
        assert!(!is_sparse(&File::open("Cargo.toml").unwrap()).unwrap());

        let dir = tempdir().unwrap();
        let (file, _) = tmps(&dir);
        create_sparse_with_data(&file, 0, 0);

        {
            let fd = File::open(&file).unwrap();
            assert!(is_sparse(&fd).unwrap());
        }
        {
            let mut fd = File::open(&file).unwrap();
            write!(fd, "{}", "test");
        }
        {
            let fd = File::open(&file).unwrap();
            assert!(is_sparse(&fd).unwrap());
        }
    }

    #[test]
    fn test_copy_range_sparse() {
        let dir = tempdir().unwrap();
        let (file, from) = tmps(&dir);
        let data = "test data";

        {
            let mut fd = File::create(&from).unwrap();
            write!(fd, "{}", data);
        }

        create_sparse(&file);

        {
            let infd = File::open(&from).unwrap();
            let outfd: File = OpenOptions::new()
                .write(true)
                .append(false)
                .open(&file).unwrap();
            copy_file_bytes(&infd, &outfd, data.len() as u64).unwrap();
        }

        assert!(is_sparse(&File::open(&file).unwrap()).unwrap());
    }

    #[test]
    fn test_sparse_copy_middle() {
        let dir = tempdir().unwrap();
        let (file, from) = tmps(&dir);
        let data = "test data";

        {
            let mut fd = File::create(&from).unwrap();
            write!(fd, "{}", data);
        }

        create_sparse(&file);

        let offset: usize = 512*1024;
        {
            let infd = File::open(&from).unwrap();
            let outfd: File = OpenOptions::new()
                .write(true)
                .append(false)
                .open(&file).unwrap();
            let mut offdat: i64 = 512*1024;
            let offptr = &mut offdat as *mut i64;
            cvt(
                unsafe {
                    copy_file_range(
                        infd.as_raw_fd(),
                        ptr::null_mut(),
                        outfd.as_raw_fd(),
                        offptr,
                        data.len(),
                        0) as i64
                }).unwrap();
        }

        assert!(is_sparse(&File::open(&file).unwrap()).unwrap());

        let bytes = read(&file).unwrap();
        assert!(bytes.len() == 1024*1024);
        assert!(bytes[offset] == b't');
        assert!(bytes[offset+1] == b'e');
        assert!(bytes[offset+2] == b's');
        assert!(bytes[offset+3] == b't');
        assert!(bytes[offset+data.len()] == 0);
    }

    #[test]
    fn test_lseek_data() {
        let dir = tempdir().unwrap();
        let (file, from) = tmps(&dir);
        let data = "test data";
        let offset = 512*1024;

        {
            let mut fd = File::create(&from).unwrap();
            write!(fd, "{}", data);
        }

        create_sparse(&file);

        {
            let infd = File::open(&from).unwrap();
            let outfd: File = OpenOptions::new()
                .write(true)
                .append(false)
                .open(&file).unwrap();
            cvt(
                unsafe {
                    copy_file_range(
                        infd.as_raw_fd(),
                        ptr::null_mut(),
                        outfd.as_raw_fd(),
                        &mut (offset as i64) as *mut i64,
                        data.len(),
                        0) as i64
                }).unwrap();
        }

        assert!(is_sparse(&File::open(&file).unwrap()).unwrap());

        let off = lseek(&File::open(&file).unwrap(), 0, Wence::Data).unwrap();
        assert_eq!(off, SeekOff::Offset(offset));
    }

    #[test]
    fn test_sparse_rust_seek() {
        let dir = tempdir().unwrap();
        let (file, _) = tmps(&dir);

        let len = create_sparse_with_data(&file, 0, 10) as usize;
        assert!(is_sparse(&File::open(&file).unwrap()).unwrap());

        let bytes = read(&file).unwrap();
        assert!(bytes.len() == len);

        let offset = 1024 * 4096;
        assert!(bytes[offset] == b'c');
        assert!(bytes[offset+1] == b'0');
        assert!(bytes[offset+2] == b'0');
        assert!(bytes[offset+3] == b'l');
    }


    #[test]
    fn test_lseek_no_data() {
        let dir = tempdir().unwrap();
        let (file, _) = tmps(&dir);
        create_sparse(&file);

        assert!(is_sparse(&File::open(&file).unwrap()).unwrap());

        let fd = File::open(&file).unwrap();
        let off = lseek(&fd, 0, Wence::Data).unwrap();
        assert!(off == SeekOff::EOF);
    }

    #[test]
    fn test_allocate_file_is_sparse() {
        let dir = tempdir().unwrap();
        let (file, _) = tmps(&dir);
        let len = 32 * 1024 * 1024;

        {
            let fd = File::create(&file).unwrap();
            allocate_file(&fd, len).unwrap();
        }

        {
            let fd = File::open(&file).unwrap();
            assert_eq!(len, fd.metadata().unwrap().len());
            assert!(is_sparse(&fd).unwrap());
        }
    }
}
