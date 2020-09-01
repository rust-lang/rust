use crate::io::{self, ErrorKind, Read, Write};
use crate::mem::MaybeUninit;

/// Copies the entire contents of a reader into a writer.
///
/// This function will continuously read data from `reader` and then
/// write it into `writer` in a streaming fashion until `reader`
/// returns EOF.
///
/// On success, the total number of bytes that were copied from
/// `reader` to `writer` is returned.
///
/// If you’re wanting to copy the contents of one file to another and you’re
/// working with filesystem paths, see the [`fs::copy`] function.
///
/// [`fs::copy`]: crate::fs::copy
///
/// # Errors
///
/// This function will return an error immediately if any call to [`read`] or
/// [`write`] returns an error. All instances of [`ErrorKind::Interrupted`] are
/// handled by this function and the underlying operation is retried.
///
/// [`read`]: Read::read
/// [`write`]: Write::write
///
/// # Examples
///
/// ```
/// use std::io;
///
/// fn main() -> io::Result<()> {
///     let mut reader: &[u8] = b"hello";
///     let mut writer: Vec<u8> = vec![];
///
///     io::copy(&mut reader, &mut writer)?;
///
///     assert_eq!(&b"hello"[..], &writer[..]);
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> io::Result<u64>
where
    R: Read,
    W: Write,
{
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        kernel_copy::copy_spec(reader, writer)
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    generic_copy(reader, writer)
}

pub(crate) fn generic_copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> io::Result<u64>
where
    R: Read,
    W: Write,
{
    let mut buf = MaybeUninit::<[u8; super::DEFAULT_BUF_SIZE]>::uninit();
    // FIXME: #42788
    //
    //   - This creates a (mut) reference to a slice of
    //     _uninitialized_ integers, which is **undefined behavior**
    //
    //   - Only the standard library gets to soundly "ignore" this,
    //     based on its privileged knowledge of unstable rustc
    //     internals;
    unsafe {
        reader.initializer().initialize(buf.assume_init_mut());
    }

    let mut written = 0;
    loop {
        let len = match reader.read(unsafe { buf.assume_init_mut() }) {
            Ok(0) => return Ok(written),
            Ok(len) => len,
            Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        writer.write_all(unsafe { &buf.assume_init_ref()[..len] })?;
        written += len as u64;
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod kernel_copy {

    use crate::cmp::min;
    use crate::convert::TryInto;
    use crate::fs::{File, Metadata};
    use crate::io::{
        BufRead, BufReader, BufWriter, Read, Result, StderrLock, StdinLock, StdoutLock, Take, Write,
    };
    use crate::mem::ManuallyDrop;
    use crate::net::TcpStream;
    use crate::os::unix::fs::FileTypeExt;
    use crate::os::unix::io::{AsRawFd, FromRawFd, RawFd};
    use crate::process::{ChildStderr, ChildStdin, ChildStdout};

    pub(super) fn copy_spec<R: Read + ?Sized, W: Write + ?Sized>(
        read: &mut R,
        write: &mut W,
    ) -> Result<u64> {
        let copier = Copier { read, write };
        SpecCopy::copy(copier)
    }

    enum FdMeta {
        Metadata(Metadata),
        Socket,
        Pipe,
        None,
    }

    impl FdMeta {
        fn is_fifo(&self) -> bool {
            match self {
                FdMeta::Metadata(meta) => meta.file_type().is_fifo(),
                FdMeta::Socket => false,
                FdMeta::Pipe => true,
                FdMeta::None => false,
            }
        }
    }

    struct CopyParams(FdMeta, Option<RawFd>);

    struct Copier<'a, 'b, R: Read + ?Sized, W: Write + ?Sized> {
        pub read: &'a mut R,
        pub write: &'b mut W,
    }

    trait SpecCopy {
        fn copy(self) -> Result<u64>;
    }

    impl<R: Read + ?Sized, W: Write + ?Sized> SpecCopy for Copier<'_, '_, R, W> {
        default fn copy(self) -> Result<u64> {
            super::generic_copy(self.read, self.write)
        }
    }

    impl<R: CopyRead, W: CopyWrite> SpecCopy for Copier<'_, '_, R, W> {
        fn copy(self) -> Result<u64> {
            let (reader, writer) = (self.read, self.write);
            let r_cfg = reader.properties();
            let w_cfg = writer.properties();

            // before direct operations  on file descriptors ensure that all source and sink buffers are emtpy
            let mut flush = || -> crate::io::Result<u64> {
                let bytes = reader.drain_to(writer, u64::MAX)?;
                writer.flush()?;
                Ok(bytes)
            };

            match (r_cfg, w_cfg) {
                (
                    CopyParams(FdMeta::Metadata(reader_meta), Some(readfd)),
                    CopyParams(FdMeta::Metadata(writer_meta), Some(writefd)),
                ) if reader_meta.is_file() && writer_meta.is_file() => {
                    let bytes_flushed = flush()?;
                    let max_write = reader.min_limit();
                    let (mut reader, mut writer) =
                        unsafe { (fd_as_file(readfd), fd_as_file(writefd)) };
                    let len = reader_meta.len();
                    crate::sys::fs::copy_regular_files(
                        &mut reader,
                        &mut writer,
                        min(len, max_write),
                    )
                    .map(|bytes_copied| bytes_copied + bytes_flushed)
                }
                (CopyParams(reader_meta, Some(readfd)), CopyParams(writer_meta, Some(writefd)))
                    if reader_meta.is_fifo() || writer_meta.is_fifo() =>
                {
                    // splice
                    let bytes_flushed = flush()?;
                    let max_write = reader.min_limit();
                    let (mut reader, mut writer) =
                        unsafe { (fd_as_file(readfd), fd_as_file(writefd)) };
                    crate::sys::fs::sendfile_splice(
                        crate::sys::fs::SpliceMode::Splice,
                        &mut reader,
                        &mut writer,
                        max_write,
                    )
                    .map(|bytes_sent| bytes_sent + bytes_flushed)
                }
                (
                    CopyParams(FdMeta::Metadata(reader_meta), Some(readfd)),
                    CopyParams(_, Some(writefd)),
                ) if reader_meta.is_file() => {
                    // try sendfile, most modern systems it should work with any target as long as the source is a mmapable file.
                    // in the rare cases where it's no supported the wrapper function will fall back to a normal copy loop
                    let bytes_flushed = flush()?;
                    let (mut reader, mut writer) =
                        unsafe { (fd_as_file(readfd), fd_as_file(writefd)) };
                    let len = reader_meta.len();
                    let max_write = reader.min_limit();
                    crate::sys::fs::sendfile_splice(
                        crate::sys::fs::SpliceMode::Sendfile,
                        &mut reader,
                        &mut writer,
                        min(len, max_write),
                    )
                    .map(|bytes_sent| bytes_sent + bytes_flushed)
                }
                _ => super::generic_copy(reader, writer),
            }
        }
    }

    #[rustc_specialization_trait]
    trait CopyRead: Read {
        fn drain_to<W: Write>(&mut self, _writer: &mut W, _limit: u64) -> Result<u64> {
            Ok(0)
        }

        /// The minimum of the limit of all `Take<_>` wrappers, `u64::MAX` otherwise.
        /// This method does not account for data `BufReader` buffers and would underreport
        /// the limit of a `Take<BufReader<Take<_>>>` type. Thus its result is only valid
        /// after draining the buffers.
        fn min_limit(&self) -> u64 {
            u64::MAX
        }

        fn properties(&self) -> CopyParams;
    }

    #[rustc_specialization_trait]
    trait CopyWrite: Write {
        fn properties(&self) -> CopyParams;
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
            CopyParams(fd_to_meta(self), Some(self.as_raw_fd()))
        }
    }

    impl CopyWrite for &File {
        fn properties(&self) -> CopyParams {
            CopyParams(fd_to_meta(*self), Some(self.as_raw_fd()))
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
            CopyParams(fd_to_meta(self), Some(self.as_raw_fd()))
        }
    }

    impl CopyWrite for StderrLock<'_> {
        fn properties(&self) -> CopyParams {
            CopyParams(fd_to_meta(self), Some(self.as_raw_fd()))
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
            Err(_) => FdMeta::None,
        }
    }

    unsafe fn fd_as_file(fd: RawFd) -> ManuallyDrop<File> {
        ManuallyDrop::new(File::from_raw_fd(fd))
    }
}
