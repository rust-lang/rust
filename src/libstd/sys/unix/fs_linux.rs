
use cell::RefCell;
use cmp;
use fs::{File, Metadata};
use io::{self, Error, ErrorKind, Read, Write};
use libc;
use mem;
use os::linux::fs::MetadataExt;
use path::Path;
use ptr;
use super::ext::io::AsRawFd;
use sys::{cvt, cvt_r};

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
enum Wence {
    Set = libc::SEEK_SET as isize,
    Cur = libc::SEEK_CUR as isize,
    End = libc::SEEK_END as isize,
    Data = libc::SEEK_DATA as isize,
    Hole = libc::SEEK_HOLE as isize,
}

#[derive(PartialEq, Debug)]
enum SeekOff {
    Offset(u64),
    EOF
}

fn lseek(fd: &File, off: i64, wence: Wence) -> io::Result<SeekOff> {
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

fn allocate_file(fd: &File, len: u64) -> io::Result<()> {
    cvt_r(|| unsafe {libc::ftruncate64(fd.as_raw_fd(), len as i64)})?;
    Ok(())
}


// Wrapper for copy_file_range(2) that defers file offset tracking to
// the underlying call. See the manpage for details.
fn copy_bytes_kernel(reader: &File, writer: &File, nbytes: usize) -> io::Result<u64> {
    unsafe {
        cvt(copy_file_range(reader.as_raw_fd(),
                            ptr::null_mut(),
                            writer.as_raw_fd(),
                            ptr::null_mut(),
                            nbytes,
                            0)
        )
    }
    .map(|v| v as u64)
}

// Slightly modified version of io::copy() that only copies a set amount of bytes.
fn copy_bytes_uspace(mut reader: &File, mut writer: &File, nbytes: usize) -> io::Result<u64> {
    const BLKSIZE: usize = 4 * 1024;  // Assume 4k blocks on disk.
    let mut buf = unsafe {
        let mut buf: [u8; BLKSIZE] = mem::uninitialized();
        reader.initializer().initialize(&mut buf);
        buf
    };

    let mut written = 0;
    while written < nbytes {
        let next = cmp::min(nbytes - written, BLKSIZE);
        let len = match reader.read(&mut buf[..next]) {
            Ok(0) => return Err(Error::new(ErrorKind::InvalidData,
                                           "Source file ended prematurely.")),
            Ok(len) => len,
            Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        writer.write_all(&buf[..len])?;
        written += len;
    }
    Ok(written as u64)
}


// Kernels prior to 4.5 don't have copy_file_range,so we store the
// availability in a thread-local flag to avoid unnecessary syscalls.
thread_local! {
    static HAS_COPY_FILE_RANGE: RefCell<bool> = RefCell::new(true);
}

fn copy_bytes(reader: &File, writer: &File, uspace: bool, nbytes: u64) -> io::Result<u64> {
    HAS_COPY_FILE_RANGE.with(|cfr| {
        loop {
            if uspace || !*cfr.borrow() {
                return copy_bytes_uspace(reader, writer, nbytes as usize)

            } else {
                let result = copy_bytes_kernel(reader, writer, nbytes as usize);

                if let Err(ref err) = result {
                    match err.raw_os_error() {
                        Some(libc::ENOSYS) | Some(libc::EPERM) => {
                            // Flag as unavailable and retry.
                            *cfr.borrow_mut() = false;
                            continue;
                        }
                        _ => {}

                    }
                }
                return result;
            }
        }
    })
}


/// Copy len bytes from whereever the descriptor cursors are set.
fn copy_range(infd: &File, outfd: &File, uspace: bool, len: u64) -> io::Result<u64> {
    let mut written = 0;
    while written < len {
        let result = copy_bytes(&infd, &outfd, uspace, len - written)?;
        written += result;
    }
    Ok(written)
}

fn next_sparse_segments(fd: &File, pos: u64, len: u64) -> io::Result<(u64, u64)> {
    let next_data = match lseek(fd, pos as i64, Wence::Data)? {
        SeekOff::Offset(off) => off,
        SeekOff::EOF => len
    };
    let next_hole = match lseek(fd, next_data as i64, Wence::Hole)? {
        SeekOff::Offset(off) => off,
        SeekOff::EOF => len
    };

    Ok((next_data, next_hole))
}

fn copy_sparse(infd: &File, outfd: &File, uspace: bool, len: u64) -> io::Result<u64> {
    allocate_file(&outfd, len)?;

    let mut pos = 0;

    while pos < len {
        let (next_data, next_hole) = next_sparse_segments(infd, pos, len)?;
        lseek(infd, next_data as i64, Wence::Set)?;
        lseek(outfd, next_data as i64, Wence::Set)?;

        let _written = copy_range(infd, outfd, uspace, next_hole - next_data)?;
        pos = next_hole;
    }

    Ok(len)
}


fn copy_parms(in_meta: &Metadata, out_meta: &Metadata) -> io::Result<(bool, bool)> {
    let is_sparse = in_meta.st_blocks() < in_meta.st_size() / in_meta.st_blksize();
    let is_xmount = in_meta.st_dev() != out_meta.st_dev();
    Ok((is_sparse, is_xmount))
}


pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    if !from.is_file() {
        return Err(Error::new(ErrorKind::InvalidInput,
                              "the source path is not an existing regular file"))
    }

    let infd = File::open(from)?;
    let outfd = File::create(to)?;
    let in_meta = infd.metadata()?;
    let out_meta = outfd.metadata()?;

    let (is_sparse, is_xmount) = copy_parms(&in_meta, &out_meta)?;
    let uspace = is_xmount;

        let len = in_meta.len();
    let total = if is_sparse {
        copy_sparse(&infd, &outfd, uspace, len)?

    } else {
        copy_range(&infd, &outfd, uspace, len)?
    };

    outfd.set_permissions(in_meta.permissions())?;
    Ok(total)
}


#[cfg(test)]
mod tests {
    use super::*;
    use iter;
    use ffi::CStr;
    use sys_common::io::test::{TempDir, tmpdir};
    use fs::{read, OpenOptions};
    use io::{Seek, SeekFrom, Write};
    use path::PathBuf;

    fn supported_kernel() -> bool {
        let mut uname = unsafe { mem::zeroed() };
        cvt(unsafe { libc::uname(&mut uname) }).unwrap();

        let release =
            unsafe { CStr::from_ptr(uname.release.as_ptr()) }
            .to_str().unwrap();
        let s = release
            .split(|c| c == '.' || c == '-')
            .collect::<Vec<&str>>();

        // Kernel >= 4.5
        s[0].parse::<u32>().unwrap() >= 4 &&
            s[1].parse::<u32>().unwrap() >= 5
    }

    fn create_sparse(file: &PathBuf, len: u64) {
        let fd = File::create(file).unwrap();
        cvt(unsafe {libc::ftruncate64(fd.as_raw_fd(), len as i64)}).unwrap();
    }

    fn create_sparse_with_data(file: &PathBuf, head: u64, tail: u64) -> u64 {
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
        write!(fd, "{}", data).unwrap();

        fd.seek(SeekFrom::Start(1024*4096)).unwrap();
        write!(fd, "{}", data).unwrap();

        fd.seek(SeekFrom::Start(4096*4096)).unwrap();
        write!(fd, "{}", data).unwrap();

        len
    }


    fn tmps(dir: &TempDir) -> (PathBuf, PathBuf) {
        let from = dir.path().join("from.bin");
        let to = dir.path().join("to.bin");
        (from, to)
    }


    fn is_sparse(fd: &File) -> io::Result<bool> {
        let stat = fd.metadata()?;
        Ok(stat.st_blocks() < stat.st_size() / stat.st_blksize())
    }

    fn is_fsparse(file: &PathBuf) -> io::Result<bool> {
        let fd = File::open(file)?;
        is_sparse(&fd)
    }

    #[test]
    fn test_sparse_detection() {
        assert!(!is_sparse(&File::open("Cargo.toml").unwrap()).unwrap());

        let dir = tmpdir();
        let (from, _) = tmps(&dir);
        create_sparse_with_data(&from, 0, 0);

        {
            let fd = File::open(&from).unwrap();
            assert!(is_sparse(&fd).unwrap());
        }
        {
            let mut fd = OpenOptions::new()
                .write(true)
                .append(true)
                .open(&from).unwrap();
            write!(fd, "{}", "test").unwrap();
        }
        {
            let fd = File::open(&from).unwrap();
            assert!(is_sparse(&fd).unwrap());
        }
    }

    fn test_copy_range(uspace: bool) {
        let dir = tmpdir();
        let (from, to) = tmps(&dir);
        let data = "test data";

        {
            let mut fd = File::create(&to).unwrap();
            write!(fd, "{}", data).unwrap();
        }

        create_sparse_with_data(&from, 0, 0);

        {
            let infd = File::open(&to).unwrap();
            let outfd: File = OpenOptions::new()
                .write(true)
                .append(false)
                .open(&from).unwrap();
            copy_range(&infd, &outfd, uspace, data.len() as u64).unwrap();
        }

        assert!(is_sparse(&File::open(&from).unwrap()).unwrap());
    }

    #[test]
    fn test_copy_range_sparse_kernel() {
        test_copy_range(false);
    }

    #[test]
    fn test_copy_range_sparse_uspace() {
        test_copy_range(true);
    }

    #[test]
    fn test_sparse_copy_middle() {
        if !supported_kernel() {
            return;
        }

        let dir = tmpdir();
        let (from, to) = tmps(&dir);
        let data = "test data";

        {
            let mut fd = File::create(&to).unwrap();
            write!(fd, "{}", data).unwrap();
        }

        create_sparse(&from, 1024*1024);

        let offset = 512*1024;
        {
            let infd = File::open(&to).unwrap();
            let outfd: File = OpenOptions::new()
                .write(true)
                .append(false)
                .open(&from).unwrap();
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

        assert!(is_sparse(&File::open(&from).unwrap()).unwrap());

        let bytes = read(&from).unwrap();
        assert!(bytes.len() == 1024*1024);
        assert!(bytes[offset] == b't');
        assert!(bytes[offset+1] == b'e');
        assert!(bytes[offset+2] == b's');
        assert!(bytes[offset+3] == b't');
        assert!(bytes[offset+data.len()] == 0);
    }

    #[test]
    fn test_lseek_data() {
        if !supported_kernel() {
            return;
        }

        let dir = tmpdir();
        let (from, to) = tmps(&dir);
        let data = "test data";
        let offset = 512*1024;

        {
            let mut fd = File::create(&to).unwrap();
            write!(fd, "{}", data).unwrap();
        }

        create_sparse(&from, 1024*1024);

        {
            let infd = File::open(&to).unwrap();
            let outfd: File = OpenOptions::new()
                .write(true)
                .append(false)
                .open(&from).unwrap();
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

        assert!(is_sparse(&File::open(&from).unwrap()).unwrap());

        let off = lseek(&File::open(&from).unwrap(), 0, Wence::Data).unwrap();
        assert_eq!(off, SeekOff::Offset(offset));
    }

    #[test]
    fn test_sparse_rust_seek() {
        let dir = tmpdir();
        let (from, _) = tmps(&dir);

        let len = create_sparse_with_data(&from, 0, 10);
        assert!(is_sparse(&File::open(&from).unwrap()).unwrap());

        let bytes = read(&from).unwrap();
        assert!(bytes.len() == len as usize);

        let offset = 1024 * 4096;
        assert!(bytes[offset] == b'c');
        assert!(bytes[offset+1] == b'0');
        assert!(bytes[offset+2] == b'0');
        assert!(bytes[offset+3] == b'l');
    }


    #[test]
    fn test_lseek_no_data() {
        let dir = tmpdir();
        let (from, _) = tmps(&dir);
        create_sparse(&from, 1024*1024);

        assert!(is_sparse(&File::open(&from).unwrap()).unwrap());

        let fd = File::open(&from).unwrap();
        let off = lseek(&fd, 0, Wence::Data).unwrap();
        assert!(off == SeekOff::EOF);
    }

    #[test]
    fn test_allocate_file_is_sparse() {
        let dir = tmpdir();
        let (from, _) = tmps(&dir);
        let len = 32 * 1024 * 1024;

        {
            let fd = File::create(&from).unwrap();
            allocate_file(&fd, len).unwrap();
        }

        {
            let fd = File::open(&from).unwrap();
            assert_eq!(len, fd.metadata().unwrap().len());
            assert!(is_sparse(&fd).unwrap());
        }
    }


    #[test]
    fn test_copy_bytes_uspace_small() {
        let dir = tmpdir();
        let (from, to) = tmps(&dir);
        let data = "test data";
        let offset = 32;

        {
            let mut fd = File::create(&to).unwrap();
            write!(fd, "{}", data).unwrap();
        }

        create_sparse(&from, 128);
        create_sparse(&to, 128);

        {
            let mut fd: File = OpenOptions::new()
                .write(true)
                .append(false)
                .open(&from).unwrap();
            fd.seek(SeekFrom::Start(offset)).unwrap();
            write!(fd, "{}", data).unwrap();
        }

        {
            let mut infd = File::open(&from).unwrap();
            let mut outfd: File = OpenOptions::new()
                .write(true)
                .append(false)
                .open(&to).unwrap();
            infd.seek(SeekFrom::Start(offset)).unwrap();
            outfd.seek(SeekFrom::Start(offset)).unwrap();

            let written = copy_bytes_uspace(&infd, &outfd, data.len()).unwrap();
            assert_eq!(written, data.len() as u64);
        }

        {
            let from_data = read(&from).unwrap();
            let to_data = read(&to).unwrap();
            assert_eq!(from_data, to_data);
        }
    }

    #[test]
    fn test_copy_bytes_uspace_large() {
        let dir = tmpdir();
        let (from, to) = tmps(&dir);
        let size = 128*1024;
        let data = iter::repeat("X").take(size).collect::<String>();

        {
            let mut fd: File = File::create(&from).unwrap();
            write!(fd, "{}", data).unwrap();
        }

        {
            let infd = File::open(&from).unwrap();
            let outfd = File::create(&to).unwrap();
            let written = copy_bytes_uspace(&infd, &outfd, size).unwrap();

            assert_eq!(written, size as u64);
        }

        assert_eq!(from.metadata().unwrap().len(),
                   to.metadata().unwrap().len());

        {
            let from_data = read(&from).unwrap();
            let to_data = read(&to).unwrap();
            assert_eq!(from_data, to_data);
        }
    }




    #[test]
    fn test_simple_copy() {
        let dir = tmpdir();
        let (from, to) = tmps(&dir);
        let text = "This is a test file.";

        {
            let file = File::create(&from).unwrap();
            write!(&file, "{}", text).unwrap();
        }

        let written = copy(&from, &to).unwrap();
        assert_eq!(text.len() as u64, written);

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }

    #[test]
    fn test_sparse() {
        let dir = tmpdir();
        let (from, to) = tmps(&dir);

        let slen = create_sparse_with_data(&from, 0, 0);
        assert_eq!(slen, from.metadata().unwrap().len());
        assert!(is_fsparse(&from).unwrap());

        let written = copy(&from, &to).unwrap();
        assert_eq!(slen, written);
        assert!(is_fsparse(&to).unwrap());

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }

    #[test]
    fn test_sparse_leading_gap() {
        let dir = tmpdir();
        let (from, to) = tmps(&dir);

        let slen = create_sparse_with_data(&from, 1024, 0);
        assert_eq!(slen, from.metadata().unwrap().len());
        assert!(is_fsparse(&from).unwrap());

        let written = copy(&from, &to).unwrap();
        assert_eq!(slen, written);
        assert!(is_fsparse(&to).unwrap());

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }

    #[test]
    fn test_sparse_trailng_gap() {
        let dir = tmpdir();
        let (from, to) = tmps(&dir);

        let slen = create_sparse_with_data(&from, 1024, 1024);
        assert_eq!(slen, from.metadata().unwrap().len());
        assert!(is_fsparse(&from).unwrap());

        let written = copy(&from, &to).unwrap();
        assert_eq!(slen, written);
        assert!(is_fsparse(&to).unwrap());

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }

    #[test]
    fn test_empty_sparse() {
        let dir = tmpdir();
        let (from, to) = tmps(&dir);

        create_sparse(&from, 1024*1024);
        assert_eq!(from.metadata().unwrap().len(), 1024*1024);

        let _written = copy(&from, &to).unwrap();
        assert_eq!(to.metadata().unwrap().len(), 1024*1024);

        assert!(is_fsparse(&to).unwrap());

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }
}
