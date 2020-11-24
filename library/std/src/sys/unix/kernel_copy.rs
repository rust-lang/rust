//! This module contains specializations that can offload `io::copy()` operations on file descriptor
//! containing types (`File`, `TcpStream`, etc.) to more efficient syscalls than `read(2)` and `write(2)`.
//!
//! Specialization is only applied to wholly std-owned types so that user code can't observe
//! that the `Read` and `Write` traits are not used.
//!
//! Since a copy operation involves a reader and writer side where each can consist of different types
//! and also involve generic wrappers (e.g. `Take`, `BufReader`) it is not practical to specialize
//! a single method on all possible combinations.
//!
//! Instead readers and writers are handled separately by the `CopyRead` and `CopyWrite` specialization
//! traits and then specialized on by the `Copier::copy` method.
//!
//! `Copier` uses the specialization traits to unpack the underlying file descriptors and
//! additional prerequisites and constraints imposed by the wrapper types.
//!
//! Once it has obtained all necessary pieces and brought any wrapper types into a state where they
//! can be safely bypassed it will attempt to use the `copy_file_range(2)`,
//! `sendfile(2)` or `splice(2)` syscalls to move data directly between file descriptors.
//! Since those syscalls have requirements that cannot be fully checked in advance and
//! gathering additional information about file descriptors would require additional syscalls
//! anyway it simply attempts to use them one after another (guided by inaccurate hints) to
//! figure out which one works and and falls back to the generic read-write copy loop if none of them
//! does.
//! Once a working syscall is found for a pair of file descriptors it will be called in a loop
//! until the copy operation is completed.
//!
//! Advantages of using these syscalls:
//!
//! * fewer context switches since reads and writes are coalesced into a single syscall
//!   and more bytes are transferred per syscall. This translates to higher throughput
//!   and fewer CPU cycles, at least for sufficiently large transfers to amortize the initial probing.
//! * `copy_file_range` creates reflink copies on CoW filesystems, thus moving less data and
//!   consuming less disk space
//! * `sendfile` and `splice` can perform zero-copy IO under some circumstances while
//!   a naive copy loop would move every byte through the CPU.
//!
//! Drawbacks:
//!
//! * copy operations smaller than the default buffer size can under some circumstances, especially
//!   on older kernels, incur more syscalls than the naive approach would. As mentioned above
//!   the syscall selection is guided by hints to minimize this possibility but they are not perfect.
//! * optimizations only apply to std types. If a user adds a custom wrapper type, e.g. to report
//!   progress, they can hit a performance cliff.
//! * complexity

use crate::cmp::min;
use crate::convert::TryInto;
use crate::fs::{File, Metadata};
use crate::io::copy::generic_copy;
use crate::io::{
    BufRead, BufReader, BufWriter, Error, Read, Result, StderrLock, StdinLock, StdoutLock, Take,
    Write,
};
use crate::mem::ManuallyDrop;
use crate::net::TcpStream;
use crate::os::unix::fs::FileTypeExt;
use crate::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use crate::process::{ChildStderr, ChildStdin, ChildStdout};
use crate::ptr;
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sys::cvt;

#[cfg(test)]
mod tests;

pub(crate) fn copy_spec<R: Read + ?Sized, W: Write + ?Sized>(
    read: &mut R,
    write: &mut W,
) -> Result<u64> {
    let copier = Copier { read, write };
    SpecCopy::copy(copier)
}

/// This type represents either the inferred `FileType` of a `RawFd` based on the source
/// type from which it was extracted or the actual metadata
///
/// The methods on this type only provide hints, due to `AsRawFd` and `FromRawFd` the inferred
/// type may be wrong.
enum FdMeta {
    /// We obtained the FD from a type that can contain any type of `FileType` and queried the metadata
    /// because it is cheaper than probing all possible syscalls (reader side)
    Metadata(Metadata),
    Socket,
    Pipe,
    /// We don't have any metadata, e.g. because the original type was `File` which can represent
    /// any `FileType` and we did not query the metadata either since it did not seem beneficial
    /// (writer side)
    NoneObtained,
}

impl FdMeta {
    fn maybe_fifo(&self) -> bool {
        match self {
            FdMeta::Metadata(meta) => meta.file_type().is_fifo(),
            FdMeta::Socket => false,
            FdMeta::Pipe => true,
            FdMeta::NoneObtained => true,
        }
    }

    fn potential_sendfile_source(&self) -> bool {
        match self {
            // procfs erronously shows 0 length on non-empty readable files.
            // and if a file is truly empty then a `read` syscall will determine that and skip the write syscall
            // thus there would be benefit from attempting sendfile
            FdMeta::Metadata(meta)
                if meta.file_type().is_file() && meta.len() > 0
                    || meta.file_type().is_block_device() =>
            {
                true
            }
            _ => false,
        }
    }

    fn copy_file_range_candidate(&self) -> bool {
        match self {
            // copy_file_range will fail on empty procfs files. `read` can determine whether EOF has been reached
            // without extra cost and skip the write, thus there is no benefit in attempting copy_file_range
            FdMeta::Metadata(meta) if meta.is_file() && meta.len() > 0 => true,
            FdMeta::NoneObtained => true,
            _ => false,
        }
    }
}

struct CopyParams(FdMeta, Option<RawFd>);

struct Copier<'a, 'b, R: Read + ?Sized, W: Write + ?Sized> {
    read: &'a mut R,
    write: &'b mut W,
}

trait SpecCopy {
    fn copy(self) -> Result<u64>;
}

impl<R: Read + ?Sized, W: Write + ?Sized> SpecCopy for Copier<'_, '_, R, W> {
    default fn copy(self) -> Result<u64> {
        generic_copy(self.read, self.write)
    }
}

impl<R: CopyRead, W: CopyWrite> SpecCopy for Copier<'_, '_, R, W> {
    fn copy(self) -> Result<u64> {
        let (reader, writer) = (self.read, self.write);
        let r_cfg = reader.properties();
        let w_cfg = writer.properties();

        // before direct operations on file descriptors ensure that all source and sink buffers are empty
        let mut flush = || -> crate::io::Result<u64> {
            let bytes = reader.drain_to(writer, u64::MAX)?;
            // BufWriter buffered bytes have already been accounted for in earlier write() calls
            writer.flush()?;
            Ok(bytes)
        };

        let mut written = 0u64;

        if let (CopyParams(input_meta, Some(readfd)), CopyParams(output_meta, Some(writefd))) =
            (r_cfg, w_cfg)
        {
            written += flush()?;
            let max_write = reader.min_limit();

            if input_meta.copy_file_range_candidate() && output_meta.copy_file_range_candidate() {
                let result = copy_regular_files(readfd, writefd, max_write);

                match result {
                    CopyResult::Ended(Ok(bytes_copied)) => return Ok(bytes_copied + written),
                    CopyResult::Ended(err) => return err,
                    CopyResult::Fallback(bytes) => written += bytes,
                }
            }

            // on modern kernels sendfile can copy from any mmapable type (some but not all regular files and block devices)
            // to any writable file descriptor. On older kernels the writer side can only be a socket.
            // So we just try and fallback if needed.
            // If current file offsets + write sizes overflow it may also fail, we do not try to fix that and instead
            // fall back to the generic copy loop.
            if input_meta.potential_sendfile_source() {
                let result = sendfile_splice(SpliceMode::Sendfile, readfd, writefd, max_write);

                match result {
                    CopyResult::Ended(Ok(bytes_copied)) => return Ok(bytes_copied + written),
                    CopyResult::Ended(err) => return err,
                    CopyResult::Fallback(bytes) => written += bytes,
                }
            }

            if input_meta.maybe_fifo() || output_meta.maybe_fifo() {
                let result = sendfile_splice(SpliceMode::Splice, readfd, writefd, max_write);

                match result {
                    CopyResult::Ended(Ok(bytes_copied)) => return Ok(bytes_copied + written),
                    CopyResult::Ended(err) => return err,
                    CopyResult::Fallback(0) => { /* use the fallback below */ }
                    CopyResult::Fallback(_) => {
                        unreachable!("splice should not return > 0 bytes on the fallback path")
                    }
                }
            }
        }

        // fallback if none of the more specialized syscalls wants to work with these file descriptors
        match generic_copy(reader, writer) {
            Ok(bytes) => Ok(bytes + written),
            err => err,
        }
    }
}

#[rustc_specialization_trait]
trait CopyRead: Read {
    /// Implementations that contain buffers (i.e. `BufReader`) must transfer data from their internal
    /// buffers into `writer` until either the buffers are emptied or `limit` bytes have been
    /// transferred, whichever occurs sooner.
    /// If nested buffers are present the outer buffers must be drained first.
    ///
    /// This is necessary to directly bypass the wrapper types while preserving the data order
    /// when operating directly on the underlying file descriptors.
    fn drain_to<W: Write>(&mut self, _writer: &mut W, _limit: u64) -> Result<u64> {
        Ok(0)
    }

    /// The minimum of the limit of all `Take<_>` wrappers, `u64::MAX` otherwise.
    /// This method does not account for data `BufReader` buffers and would underreport
    /// the limit of a `Take<BufReader<Take<_>>>` type. Thus its result is only valid
    /// after draining the buffers via `drain_to`.
    fn min_limit(&self) -> u64 {
        u64::MAX
    }

    /// Extracts the file descriptor and hints/metadata, delegating through wrappers if necessary.
    fn properties(&self) -> CopyParams;
}

#[rustc_specialization_trait]
trait CopyWrite: Write {
    /// Extracts the file descriptor and hints/metadata, delegating through wrappers if necessary.
    fn properties(&self) -> CopyParams;
}

impl<T> CopyRead for &mut T
where
    T: CopyRead,
{
    fn drain_to<W: Write>(&mut self, writer: &mut W, limit: u64) -> Result<u64> {
        (**self).drain_to(writer, limit)
    }

    fn min_limit(&self) -> u64 {
        (**self).min_limit()
    }

    fn properties(&self) -> CopyParams {
        (**self).properties()
    }
}

impl<T> CopyWrite for &mut T
where
    T: CopyWrite,
{
    fn properties(&self) -> CopyParams {
        (**self).properties()
    }
}

impl CopyRead for File {
    fn properties(&self) -> CopyParams {
        CopyParams(fd_to_meta(self), Some(self.as_raw_fd()))
    }
}

impl CopyRead for &File {
    fn properties(&self) -> CopyParams {
        CopyParams(fd_to_meta(*self), Some(self.as_raw_fd()))
    }
}

impl CopyWrite for File {
    fn properties(&self) -> CopyParams {
        CopyParams(FdMeta::NoneObtained, Some(self.as_raw_fd()))
    }
}

impl CopyWrite for &File {
    fn properties(&self) -> CopyParams {
        CopyParams(FdMeta::NoneObtained, Some(self.as_raw_fd()))
    }
}

impl CopyRead for TcpStream {
    fn properties(&self) -> CopyParams {
        // avoid the stat syscall since we can be fairly sure it's a socket
        CopyParams(FdMeta::Socket, Some(self.as_raw_fd()))
    }
}

impl CopyRead for &TcpStream {
    fn properties(&self) -> CopyParams {
        // avoid the stat syscall since we can be fairly sure it's a socket
        CopyParams(FdMeta::Socket, Some(self.as_raw_fd()))
    }
}

impl CopyWrite for TcpStream {
    fn properties(&self) -> CopyParams {
        // avoid the stat syscall since we can be fairly sure it's a socket
        CopyParams(FdMeta::Socket, Some(self.as_raw_fd()))
    }
}

impl CopyWrite for &TcpStream {
    fn properties(&self) -> CopyParams {
        // avoid the stat syscall since we can be fairly sure it's a socket
        CopyParams(FdMeta::Socket, Some(self.as_raw_fd()))
    }
}

impl CopyWrite for ChildStdin {
    fn properties(&self) -> CopyParams {
        CopyParams(FdMeta::Pipe, Some(self.as_raw_fd()))
    }
}

impl CopyRead for ChildStdout {
    fn properties(&self) -> CopyParams {
        CopyParams(FdMeta::Pipe, Some(self.as_raw_fd()))
    }
}

impl CopyRead for ChildStderr {
    fn properties(&self) -> CopyParams {
        CopyParams(FdMeta::Pipe, Some(self.as_raw_fd()))
    }
}

impl CopyRead for StdinLock<'_> {
    fn drain_to<W: Write>(&mut self, writer: &mut W, outer_limit: u64) -> Result<u64> {
        let buf_reader = self.as_mut_buf();
        let buf = buf_reader.buffer();
        let buf = &buf[0..min(buf.len(), outer_limit.try_into().unwrap_or(usize::MAX))];
        let bytes_drained = buf.len();
        writer.write_all(buf)?;
        buf_reader.consume(bytes_drained);

        Ok(bytes_drained as u64)
    }

    fn properties(&self) -> CopyParams {
        CopyParams(fd_to_meta(self), Some(self.as_raw_fd()))
    }
}

impl CopyWrite for StdoutLock<'_> {
    fn properties(&self) -> CopyParams {
        CopyParams(FdMeta::NoneObtained, Some(self.as_raw_fd()))
    }
}

impl CopyWrite for StderrLock<'_> {
    fn properties(&self) -> CopyParams {
        CopyParams(FdMeta::NoneObtained, Some(self.as_raw_fd()))
    }
}

impl<T: CopyRead> CopyRead for Take<T> {
    fn drain_to<W: Write>(&mut self, writer: &mut W, outer_limit: u64) -> Result<u64> {
        let local_limit = self.limit();
        let combined_limit = min(outer_limit, local_limit);
        let bytes_drained = self.get_mut().drain_to(writer, combined_limit)?;
        // update limit since read() was bypassed
        self.set_limit(local_limit - bytes_drained);

        Ok(bytes_drained)
    }

    fn min_limit(&self) -> u64 {
        min(Take::limit(self), self.get_ref().min_limit())
    }

    fn properties(&self) -> CopyParams {
        self.get_ref().properties()
    }
}

impl<T: CopyRead> CopyRead for BufReader<T> {
    fn drain_to<W: Write>(&mut self, writer: &mut W, outer_limit: u64) -> Result<u64> {
        let buf = self.buffer();
        let buf = &buf[0..min(buf.len(), outer_limit.try_into().unwrap_or(usize::MAX))];
        let bytes = buf.len();
        writer.write_all(buf)?;
        self.consume(bytes);

        let remaining = outer_limit - bytes as u64;

        // in case of nested bufreaders we also need to drain the ones closer to the source
        let inner_bytes = self.get_mut().drain_to(writer, remaining)?;

        Ok(bytes as u64 + inner_bytes)
    }

    fn min_limit(&self) -> u64 {
        self.get_ref().min_limit()
    }

    fn properties(&self) -> CopyParams {
        self.get_ref().properties()
    }
}

impl<T: CopyWrite> CopyWrite for BufWriter<T> {
    fn properties(&self) -> CopyParams {
        self.get_ref().properties()
    }
}

fn fd_to_meta<T: AsRawFd>(fd: &T) -> FdMeta {
    let fd = fd.as_raw_fd();
    let file: ManuallyDrop<File> = ManuallyDrop::new(unsafe { File::from_raw_fd(fd) });
    match file.metadata() {
        Ok(meta) => FdMeta::Metadata(meta),
        Err(_) => FdMeta::NoneObtained,
    }
}

pub(super) enum CopyResult {
    Ended(Result<u64>),
    Fallback(u64),
}

/// linux-specific implementation that will attempt to use copy_file_range for copy offloading
/// as the name says, it only works on regular files
///
/// Callers must handle fallback to a generic copy loop.
/// `Fallback` may indicate non-zero number of bytes already written
/// if one of the files' cursor +`max_len` would exceed u64::MAX (`EOVERFLOW`).
pub(super) fn copy_regular_files(reader: RawFd, writer: RawFd, max_len: u64) -> CopyResult {
    use crate::cmp;

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
        libc::syscall(libc::SYS_copy_file_range, fd_in, off_in, fd_out, off_out, len, flags)
    }

    let has_copy_file_range = HAS_COPY_FILE_RANGE.load(Ordering::Relaxed);
    let mut written = 0u64;
    while written < max_len {
        let copy_result = if has_copy_file_range {
            let bytes_to_copy = cmp::min(max_len - written, usize::MAX as u64);
            // cap to 1GB chunks in case u64::MAX is passed as max_len and the file has a non-zero seek position
            // this allows us to copy large chunks without hitting EOVERFLOW,
            // unless someone sets a file offset close to u64::MAX - 1GB, in which case a fallback would be required
            let bytes_to_copy = cmp::min(bytes_to_copy as usize, 0x4000_0000usize);
            let copy_result = unsafe {
                // We actually don't have to adjust the offsets,
                // because copy_file_range adjusts the file offset automatically
                cvt(copy_file_range(
                    reader,
                    ptr::null_mut(),
                    writer,
                    ptr::null_mut(),
                    bytes_to_copy,
                    0,
                ))
            };
            if let Err(ref copy_err) = copy_result {
                match copy_err.raw_os_error() {
                    Some(libc::ENOSYS | libc::EPERM | libc::EOPNOTSUPP) => {
                        HAS_COPY_FILE_RANGE.store(false, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
            copy_result
        } else {
            Err(Error::from_raw_os_error(libc::ENOSYS))
        };
        match copy_result {
            Ok(0) if written == 0 => {
                // fallback to work around several kernel bugs where copy_file_range will fail to
                // copy any bytes and return 0 instead of an error if
                // - reading virtual files from the proc filesystem which appear to have 0 size
                //   but are not empty. noted in coreutils to affect kernels at least up to 5.6.19.
                // - copying from an overlay filesystem in docker. reported to occur on fedora 32.
                return CopyResult::Fallback(0);
            }
            Ok(0) => return CopyResult::Ended(Ok(written)), // reached EOF
            Ok(ret) => written += ret as u64,
            Err(err) => {
                return match err.raw_os_error() {
                    // when file offset + max_length > u64::MAX
                    Some(libc::EOVERFLOW) => CopyResult::Fallback(written),
                    Some(
                        libc::ENOSYS | libc::EXDEV | libc::EINVAL | libc::EPERM | libc::EOPNOTSUPP,
                    ) => {
                        // Try fallback io::copy if either:
                        // - Kernel version is < 4.5 (ENOSYS)
                        // - Files are mounted on different fs (EXDEV)
                        // - copy_file_range is broken in various ways on RHEL/CentOS 7 (EOPNOTSUPP)
                        // - copy_file_range is disallowed, for example by seccomp (EPERM)
                        // - copy_file_range cannot be used with pipes or device nodes (EINVAL)
                        assert_eq!(written, 0);
                        CopyResult::Fallback(0)
                    }
                    _ => CopyResult::Ended(Err(err)),
                };
            }
        }
    }
    CopyResult::Ended(Ok(written))
}

#[derive(PartialEq)]
enum SpliceMode {
    Sendfile,
    Splice,
}

/// performs splice or sendfile between file descriptors
/// Does _not_ fall back to a generic copy loop.
fn sendfile_splice(mode: SpliceMode, reader: RawFd, writer: RawFd, len: u64) -> CopyResult {
    static HAS_SENDFILE: AtomicBool = AtomicBool::new(true);
    static HAS_SPLICE: AtomicBool = AtomicBool::new(true);

    syscall! {
        fn splice(
            srcfd: libc::c_int,
            src_offset: *const i64,
            dstfd: libc::c_int,
            dst_offset: *const i64,
            len: libc::size_t,
            flags: libc::c_int
        ) -> libc::ssize_t
    }

    match mode {
        SpliceMode::Sendfile if !HAS_SENDFILE.load(Ordering::Relaxed) => {
            return CopyResult::Fallback(0);
        }
        SpliceMode::Splice if !HAS_SPLICE.load(Ordering::Relaxed) => {
            return CopyResult::Fallback(0);
        }
        _ => (),
    }

    let mut written = 0u64;
    while written < len {
        // according to its manpage that's the maximum size sendfile() will copy per invocation
        let chunk_size = crate::cmp::min(len - written, 0x7ffff000_u64) as usize;

        let result = match mode {
            SpliceMode::Sendfile => {
                cvt(unsafe { libc::sendfile(writer, reader, ptr::null_mut(), chunk_size) })
            }
            SpliceMode::Splice => cvt(unsafe {
                splice(reader, ptr::null_mut(), writer, ptr::null_mut(), chunk_size, 0)
            }),
        };

        match result {
            Ok(0) => break, // EOF
            Ok(ret) => written += ret as u64,
            Err(err) => {
                return match err.raw_os_error() {
                    Some(libc::ENOSYS | libc::EPERM) => {
                        // syscall not supported (ENOSYS)
                        // syscall is disallowed, e.g. by seccomp (EPERM)
                        match mode {
                            SpliceMode::Sendfile => HAS_SENDFILE.store(false, Ordering::Relaxed),
                            SpliceMode::Splice => HAS_SPLICE.store(false, Ordering::Relaxed),
                        }
                        assert_eq!(written, 0);
                        CopyResult::Fallback(0)
                    }
                    Some(libc::EINVAL) => {
                        // splice/sendfile do not support this particular file descriptor (EINVAL)
                        assert_eq!(written, 0);
                        CopyResult::Fallback(0)
                    }
                    Some(os_err) if mode == SpliceMode::Sendfile && os_err == libc::EOVERFLOW => {
                        CopyResult::Fallback(written)
                    }
                    _ => CopyResult::Ended(Err(err)),
                };
            }
        }
    }
    CopyResult::Ended(Ok(written))
}
