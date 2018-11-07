
use io::{self, Error, ErrorKind};
use libc;
use mem;
use path::Path;
use ptr;
use sys::cvt;
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
