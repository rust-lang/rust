use std::any::Any;
use std::collections::BTreeMap;
use std::fs::{Dir, File};
use std::io::{ErrorKind, IsTerminal, Read, Seek, SeekFrom, Write};
use std::marker::CoercePointee;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::{fs, io};

use rustc_abi::Size;

use crate::shims::unix::UnixFileDescription;
use crate::*;

/// A unique id for file descriptions. While we could use the address, considering that
/// is definitely unique, the address would expose interpreter internal state when used
/// for sorting things. So instead we generate a unique id per file description which is the same
/// for all `dup`licates and is never reused.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct FdId(usize);

impl FdId {
    pub fn to_usize(self) -> usize {
        self.0
    }

    /// Create a new fd id from a `usize` without checking if this fd exists.
    pub fn new_unchecked(id: usize) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone)]
struct FdIdWith<T: ?Sized> {
    id: FdId,
    inner: T,
}

/// A refcounted pointer to a file description, also tracking the
/// globally unique ID of this file description.
#[repr(transparent)]
#[derive(CoercePointee, Debug)]
// Sadly `CoercePointee` does not let us keep the `FdId` *outside* the `Rc`.
pub struct FileDescriptionRef<T: ?Sized>(Rc<FdIdWith<T>>);

impl<T: ?Sized> Clone for FileDescriptionRef<T> {
    fn clone(&self) -> Self {
        FileDescriptionRef(self.0.clone())
    }
}

impl<T: ?Sized> Deref for FileDescriptionRef<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.inner
    }
}

impl<T: ?Sized> FileDescriptionRef<T> {
    pub fn id(&self) -> FdId {
        self.0.id
    }
}

impl<T: ?Sized> PartialEq for FileDescriptionRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl<T: ?Sized> Eq for FileDescriptionRef<T> {}

/// Holds a weak reference to the actual file description.
#[derive(Debug)]
pub struct WeakFileDescriptionRef<T: ?Sized>(Weak<FdIdWith<T>>);

impl<T: ?Sized> Clone for WeakFileDescriptionRef<T> {
    fn clone(&self) -> Self {
        WeakFileDescriptionRef(self.0.clone())
    }
}

impl<T: ?Sized> FileDescriptionRef<T> {
    pub fn downgrade(this: &Self) -> WeakFileDescriptionRef<T> {
        WeakFileDescriptionRef(Rc::downgrade(&this.0))
    }
}

impl<T: ?Sized> WeakFileDescriptionRef<T> {
    pub fn upgrade(&self) -> Option<FileDescriptionRef<T>> {
        self.0.upgrade().map(FileDescriptionRef)
    }

    /// Returns whether the file description that this weak reference points to
    /// has been closed, i.e., there are no more strong references.
    pub fn is_closed(&self) -> bool {
        self.0.strong_count() == 0
    }
}

impl<T> VisitProvenance for FileDescriptionRef<T> {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // All our FileDescription instances do not have any provenance.
    }
}

/// A helper trait to indirectly allow downcasting on `Rc<FdIdWith<dyn _>>`.
/// Ideally we'd just add a `FdIdWith<Self>: Any` bound to the `FileDescription` trait,
/// but that does not allow upcasting.
pub trait FileDescriptionExt: 'static {
    fn into_rc_any(self: FileDescriptionRef<Self>) -> Rc<dyn Any>;
}

impl<T: FileDescription + 'static> FileDescriptionExt for T {
    fn into_rc_any(self: FileDescriptionRef<Self>) -> Rc<dyn Any> {
        self.0
    }
}

pub type DynFileDescriptionRef = FileDescriptionRef<dyn FileDescription>;
pub type WeakDynFileDescriptionRef = WeakFileDescriptionRef<dyn FileDescription>;

impl FileDescriptionRef<dyn FileDescription> {
    pub fn downcast<T: FileDescription + 'static>(self) -> Option<FileDescriptionRef<T>> {
        let inner = self.into_rc_any().downcast::<FdIdWith<T>>().ok()?;
        Some(FileDescriptionRef(inner))
    }
}

/// Represents an open file description.
pub trait FileDescription: std::fmt::Debug + FileDescriptionExt {
    fn name(&self) -> &'static str;

    /// Reads as much as possible into the given buffer `ptr`.
    /// `len` indicates how many bytes we should try to read.
    ///
    /// When the read is done, `finish` will be called. Note that `read` itself may return before
    /// that happens! Everything that should happen "after" the `read` needs to happen inside
    /// `finish`.
    fn read<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _ptr: Pointer,
        _len: usize,
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot read from {}", self.name());
    }

    /// Writes as much as possible from the given buffer `ptr`.
    /// `len` indicates how many bytes we should try to write.
    ///
    /// When the write is done, `finish` will be called. Note that `write` itself may return before
    /// that happens! Everything that should happen "after" the `write` needs to happen inside
    /// `finish`.
    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _ptr: Pointer,
        _len: usize,
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot write to {}", self.name());
    }

    /// Determines whether this FD non-deterministically has its reads and writes shortened.
    fn short_fd_operations(&self) -> bool {
        // We only enable this for FD kinds where we think short accesses gain useful test coverage.
        false
    }

    /// Seeks to the given offset (which can be relative to the beginning, end, or current position).
    /// Returns the new position from the start of the stream.
    fn seek<'tcx>(
        &self,
        _communicate_allowed: bool,
        _offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>> {
        throw_unsup_format!("cannot seek on {}", self.name());
    }

    /// Returns the metadata for this FD, if available.
    /// This is either host metadata, or a non-file-backed-FD type.
    /// The latter is for new represented as a string storing a `libc` name so we only
    /// support that kind of metadata on Unix targets.
    fn metadata<'tcx>(&self) -> InterpResult<'tcx, Either<io::Result<fs::Metadata>, &'static str>> {
        throw_unsup_format!("obtaining metadata is only supported on file-backed file descriptors");
    }

    fn is_tty(&self, _communicate_allowed: bool) -> bool {
        // Most FDs are not tty's and the consequence of a wrong `false` are minor,
        // so we use a default impl here.
        false
    }

    fn as_unix<'tcx>(
        self: FileDescriptionRef<Self>,
        _ecx: &MiriInterpCx<'tcx>,
    ) -> FileDescriptionRef<dyn UnixFileDescription> {
        panic!("Not a unix file descriptor: {}", self.name());
    }

    /// Implementation of fcntl(F_GETFL) for this FD.
    fn get_flags<'tcx>(&self, _ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, Scalar> {
        throw_unsup_format!("fcntl: {} is not supported for F_GETFL", self.name());
    }

    /// Implementation of fcntl(F_SETFL) for this FD.
    fn set_flags<'tcx>(
        &self,
        _flag: i32,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        throw_unsup_format!("fcntl: {} is not supported for F_SETFL", self.name());
    }

    /// Get the `ReadinessWatched` of the file description.
    fn readiness_watched(&self) -> Option<&ReadinessWatched> {
        None
    }

    /// Get the current I/O readiness of the file description.
    fn readiness(&self) -> Readiness {
        panic!("FD type {} implements `readiness_watched` but not `readiness`", self.name());
    }
}

impl FileDescription for io::Stdin {
    fn name(&self) -> &'static str {
        "stdin"
    }

    fn read<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        if !communicate_allowed {
            // We want isolation mode to be deterministic, so we have to disallow all reads, even stdin.
            helpers::isolation_abort_error("`read` from stdin")?;
        }

        let mut stdin = &*self;
        let result = ecx.read_from_host(|buf| stdin.read(buf), len, ptr)?;
        finish.call(ecx, result)
    }

    fn is_tty(&self, communicate_allowed: bool) -> bool {
        communicate_allowed && self.is_terminal()
    }
}

impl FileDescription for io::Stdout {
    fn name(&self) -> &'static str {
        "stdout"
    }

    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        // We allow writing to stdout even with isolation enabled.
        let result = ecx.write_to_host(&*self, len, ptr)?;
        // Stdout is buffered, flush to make sure it appears on the
        // screen.  This is the write() syscall of the interpreted
        // program, we want it to correspond to a write() syscall on
        // the host -- there is no good in adding extra buffering
        // here.
        io::stdout().flush().unwrap();

        finish.call(ecx, result)
    }

    fn is_tty(&self, communicate_allowed: bool) -> bool {
        communicate_allowed && self.is_terminal()
    }
}

impl FileDescription for io::Stderr {
    fn name(&self) -> &'static str {
        "stderr"
    }

    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        // We allow writing to stderr even with isolation enabled.
        let result = ecx.write_to_host(&*self, len, ptr)?;
        // No need to flush, stderr is not buffered.
        finish.call(ecx, result)
    }

    fn is_tty(&self, communicate_allowed: bool) -> bool {
        communicate_allowed && self.is_terminal()
    }
}

#[derive(Debug)]
pub struct FileHandle {
    pub(crate) file: File,
    pub(crate) readable: bool,
    pub(crate) writable: bool,
}

impl FileDescription for FileHandle {
    fn name(&self) -> &'static str {
        "file"
    }

    fn read<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");

        if !self.readable {
            return finish.call(ecx, Err(ErrorKind::PermissionDenied.into()));
        }

        let mut file = &self.file;
        let result = ecx.read_from_host(|buf| file.read(buf), len, ptr)?;
        finish.call(ecx, result)
    }

    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");

        if !self.writable {
            // Linux hosts return EBADF here which we can't translate via the platform-independent
            // code since it does not map to any `io::ErrorKind` -- so if we don't do anything
            // special, we'd throw an "unsupported error code" here. Windows returns something that
            // gets translated to `PermissionDenied`. That seems like a good value so let's just use
            // this everywhere, even if it means behavior on Unix targets does not match the real
            // thing.
            return finish.call(ecx, Err(ErrorKind::PermissionDenied.into()));
        }
        let result = ecx.write_to_host(&self.file, len, ptr)?;
        finish.call(ecx, result)
    }

    fn seek<'tcx>(
        &self,
        communicate_allowed: bool,
        offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        interp_ok((&mut &self.file).seek(offset))
    }

    fn metadata<'tcx>(&self) -> InterpResult<'tcx, Either<io::Result<fs::Metadata>, &'static str>> {
        interp_ok(Either::Left(self.file.metadata()))
    }

    fn is_tty(&self, communicate_allowed: bool) -> bool {
        communicate_allowed && self.file.is_terminal()
    }

    fn short_fd_operations(&self) -> bool {
        // While short accesses on file-backed FDs are very rare (at least for sufficiently small
        // accesses), they can realistically happen when a signal interrupts the syscall.
        // FIXME: we should return `false` if this is a named pipe...
        true
    }

    fn as_unix<'tcx>(
        self: FileDescriptionRef<Self>,
        ecx: &MiriInterpCx<'tcx>,
    ) -> FileDescriptionRef<dyn UnixFileDescription> {
        assert!(
            ecx.target_os_is_unix(),
            "unix file operations are only available for unix targets"
        );
        self
    }
}

#[derive(Debug)]
pub struct DirHandle {
    pub(crate) dir: Dir,
}

impl FileDescription for DirHandle {
    fn name(&self) -> &'static str {
        "directory"
    }

    fn metadata<'tcx>(
        &self,
    ) -> InterpResult<'tcx, Either<io::Result<std::fs::Metadata>, &'static str>> {
        interp_ok(Either::Left(self.dir.metadata()))
    }
}

/// Like /dev/null
#[derive(Debug)]
pub struct NullOutput;

impl FileDescription for NullOutput {
    fn name(&self) -> &'static str {
        "stderr and stdout"
    }

    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        // We just don't write anything, but report to the user that we did.
        finish.call(ecx, Ok(len))
    }
}

/// Internal type of a file-descriptor - this is what [`FdTable`] expects
pub type FdNum = i32;

/// The file descriptor table
#[derive(Debug)]
pub struct FdTable {
    fds: BTreeMap<FdNum, DynFileDescriptionRef>,
    /// Unique identifier for file description, used to differentiate between various file description.
    next_file_description_id: FdId,
}

impl VisitProvenance for FdTable {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // All our FileDescription instances do not have any provenance.
    }
}

impl FdTable {
    fn new() -> Self {
        FdTable { fds: BTreeMap::new(), next_file_description_id: FdId(0) }
    }
    pub(crate) fn init(mute_stdout_stderr: bool) -> FdTable {
        let mut fds = FdTable::new();
        fds.insert_new(io::stdin());
        if mute_stdout_stderr {
            assert_eq!(fds.insert_new(NullOutput), 1);
            assert_eq!(fds.insert_new(NullOutput), 2);
        } else {
            assert_eq!(fds.insert_new(io::stdout()), 1);
            assert_eq!(fds.insert_new(io::stderr()), 2);
        }
        fds
    }

    pub fn new_ref<T: FileDescription>(&mut self, fd: T) -> FileDescriptionRef<T> {
        let file_handle =
            FileDescriptionRef(Rc::new(FdIdWith { id: self.next_file_description_id, inner: fd }));
        self.next_file_description_id = FdId(self.next_file_description_id.0.strict_add(1));
        file_handle
    }

    /// Insert a new file description to the FdTable.
    pub fn insert_new(&mut self, fd: impl FileDescription) -> FdNum {
        let fd_ref = self.new_ref(fd);
        self.insert(fd_ref)
    }

    /// Insert an alias to an existing file description to the FdTable.
    pub fn insert(&mut self, fd_ref: DynFileDescriptionRef) -> FdNum {
        self.insert_with_min_num(fd_ref, 0)
    }

    /// Insert a file description, giving it a file descriptor that is at least `min_fd_num`.
    pub fn insert_with_min_num(
        &mut self,
        file_handle: DynFileDescriptionRef,
        min_fd_num: FdNum,
    ) -> FdNum {
        let mut candidate = min_fd_num;
        for (&fd_num, _) in self.fds.range(min_fd_num..) {
            if fd_num == candidate {
                // This one is taken. Try the next one.
                candidate = candidate.strict_add(1);
            } else {
                // We found a gap! Use this candidate.
                break;
            }
        }
        // If we exhaust the loop, the table is a solid block starting at `min_fd_num` until the
        // end, and `candidate` is now the first number after that block -- exactly what we need.

        self.fds.try_insert(candidate, file_handle).unwrap();
        candidate
    }

    pub fn get(&self, fd_num: FdNum) -> Option<DynFileDescriptionRef> {
        let fd = self.fds.get(&fd_num)?;
        Some(fd.clone())
    }

    pub fn remove(&mut self, fd_num: FdNum) -> Option<DynFileDescriptionRef> {
        self.fds.remove(&fd_num)
    }

    pub fn is_fd_num(&self, fd_num: FdNum) -> bool {
        self.fds.contains_key(&fd_num)
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Read data from a host `Read` type, store the result into machine memory,
    /// and return whether that worked.
    fn read_from_host(
        &mut self,
        mut read_cb: impl FnMut(&mut [u8]) -> io::Result<usize>,
        len: usize,
        ptr: Pointer,
    ) -> InterpResult<'tcx, Result<usize, IoError>> {
        let this = self.eval_context_mut();

        let mut bytes = vec![0; len];
        let result = read_cb(&mut bytes);
        match result {
            Ok(read_size) => {
                // If reading to `bytes` did not fail, we write those bytes to the buffer.
                // Crucially, if fewer than `bytes.len()` bytes were read, only write
                // that much into the output buffer!
                this.write_bytes_ptr(ptr, bytes[..read_size].iter().copied())?;
                interp_ok(Ok(read_size))
            }
            Err(e) => interp_ok(Err(IoError::HostError(e))),
        }
    }

    /// Write data to a host `Write` type, with the bytes taken from machine memory.
    fn write_to_host(
        &mut self,
        mut file: impl io::Write,
        len: usize,
        ptr: Pointer,
    ) -> InterpResult<'tcx, Result<usize, IoError>> {
        let this = self.eval_context_mut();

        let bytes = this.read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(len))?;
        let result = file.write(bytes);
        interp_ok(result.map_err(IoError::HostError))
    }
}
