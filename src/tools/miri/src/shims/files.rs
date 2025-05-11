use std::any::Any;
use std::collections::BTreeMap;
use std::fs::{File, Metadata};
use std::io::{IsTerminal, Seek, SeekFrom, Write};
use std::marker::CoercePointee;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::{fs, io};

use rustc_abi::Size;

use crate::shims::unix::UnixFileDescription;
use crate::*;

/// A unique id for file descriptions. While we could use the address, considering that
/// is definitely unique, the address would expose interpreter internal state when used
/// for sorting things. So instead we generate a unique id per file description is the name
/// for all `dup`licates and is never reused.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct FdId(usize);

#[derive(Debug, Clone)]
struct FdIdWith<T: ?Sized> {
    id: FdId,
    inner: T,
}

/// A refcounted pointer to a file description, also tracking the
/// globally unique ID of this file description.
#[repr(transparent)]
#[derive(CoercePointee, Debug)]
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
}

impl<T> VisitProvenance for WeakFileDescriptionRef<T> {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // A weak reference can never be the only reference to some pointer or place.
        // Since the actual file description is tracked by strong ref somewhere,
        // it is ok to make this a NOP operation.
    }
}

/// A helper trait to indirectly allow downcasting on `Rc<FdIdWith<dyn _>>`.
/// Ideally we'd just add a `FdIdWith<Self>: Any` bound to the `FileDescription` trait,
/// but that does not allow upcasting.
pub trait FileDescriptionExt: 'static {
    fn into_rc_any(self: FileDescriptionRef<Self>) -> Rc<dyn Any>;

    /// We wrap the regular `close` function generically, so both handle `Rc::into_inner`
    /// and epoll interest management.
    fn close_ref<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>>;
}

impl<T: FileDescription + 'static> FileDescriptionExt for T {
    fn into_rc_any(self: FileDescriptionRef<Self>) -> Rc<dyn Any> {
        self.0
    }

    fn close_ref<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        match Rc::into_inner(self.0) {
            Some(fd) => {
                // Remove entry from the global epoll_event_interest table.
                ecx.machine.epoll_interests.remove(fd.id);

                fd.inner.close(communicate_allowed, ecx)
            }
            None => {
                // Not the last reference.
                interp_ok(Ok(()))
            }
        }
    }
}

pub type DynFileDescriptionRef = FileDescriptionRef<dyn FileDescription>;

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
    /// `dest` is where the return value should be stored: number of bytes read, or `-1` in case of error.
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
    /// `dest` is where the return value should be stored: number of bytes written, or `-1` in case of error.
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

    /// Seeks to the given offset (which can be relative to the beginning, end, or current position).
    /// Returns the new position from the start of the stream.
    fn seek<'tcx>(
        &self,
        _communicate_allowed: bool,
        _offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>> {
        throw_unsup_format!("cannot seek on {}", self.name());
    }

    /// Close the file descriptor.
    fn close<'tcx>(
        self,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>>
    where
        Self: Sized,
    {
        throw_unsup_format!("cannot close {}", self.name());
    }

    fn metadata<'tcx>(&self) -> InterpResult<'tcx, io::Result<fs::Metadata>> {
        throw_unsup_format!("obtaining metadata is only supported on file-backed file descriptors");
    }

    fn is_tty(&self, _communicate_allowed: bool) -> bool {
        // Most FDs are not tty's and the consequence of a wrong `false` are minor,
        // so we use a default impl here.
        false
    }

    fn as_unix<'tcx>(&self, _ecx: &MiriInterpCx<'tcx>) -> &dyn UnixFileDescription {
        panic!("Not a unix file descriptor: {}", self.name());
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

        let result = ecx.read_from_host(&*self, len, ptr)?;
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

        let result = ecx.read_from_host(&self.file, len, ptr)?;
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

    fn close<'tcx>(
        self,
        communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        // We sync the file if it was opened in a mode different than read-only.
        if self.writable {
            // `File::sync_all` does the checks that are done when closing a file. We do this to
            // to handle possible errors correctly.
            let result = self.file.sync_all();
            // Now we actually close the file and return the result.
            drop(self.file);
            interp_ok(result)
        } else {
            // We drop the file, this closes it but ignores any errors
            // produced when closing it. This is done because
            // `File::sync_all` cannot be done over files like
            // `/dev/urandom` which are read-only. Check
            // https://github.com/rust-lang/miri/issues/999#issuecomment-568920439
            // for a deeper discussion.
            drop(self.file);
            interp_ok(Ok(()))
        }
    }

    fn metadata<'tcx>(&self) -> InterpResult<'tcx, io::Result<Metadata>> {
        interp_ok(self.file.metadata())
    }

    fn is_tty(&self, communicate_allowed: bool) -> bool {
        communicate_allowed && self.file.is_terminal()
    }

    fn as_unix<'tcx>(&self, ecx: &MiriInterpCx<'tcx>) -> &dyn UnixFileDescription {
        assert!(
            ecx.target_os_is_unix(),
            "unix file operations are only available for unix targets"
        );
        self
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
    pub fds: BTreeMap<FdNum, DynFileDescriptionRef>,
    /// Unique identifier for file description, used to differentiate between various file description.
    next_file_description_id: FdId,
}

impl VisitProvenance for FdTable {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // All our FileDescription instances do not have any tags.
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

    pub fn insert(&mut self, fd_ref: DynFileDescriptionRef) -> FdNum {
        self.insert_with_min_num(fd_ref, 0)
    }

    /// Insert a file description, giving it a file descriptor that is at least `min_fd_num`.
    pub fn insert_with_min_num(
        &mut self,
        file_handle: DynFileDescriptionRef,
        min_fd_num: FdNum,
    ) -> FdNum {
        // Find the lowest unused FD, starting from min_fd. If the first such unused FD is in
        // between used FDs, the find_map combinator will return it. If the first such unused FD
        // is after all other used FDs, the find_map combinator will return None, and we will use
        // the FD following the greatest FD thus far.
        let candidate_new_fd =
            self.fds.range(min_fd_num..).zip(min_fd_num..).find_map(|((fd_num, _fd), counter)| {
                if *fd_num != counter {
                    // There was a gap in the fds stored, return the first unused one
                    // (note that this relies on BTreeMap iterating in key order)
                    Some(counter)
                } else {
                    // This fd is used, keep going
                    None
                }
            });
        let new_fd_num = candidate_new_fd.unwrap_or_else(|| {
            // find_map ran out of BTreeMap entries before finding a free fd, use one plus the
            // maximum fd in the map
            self.fds.last_key_value().map(|(fd_num, _)| fd_num.strict_add(1)).unwrap_or(min_fd_num)
        });

        self.fds.try_insert(new_fd_num, file_handle).unwrap();
        new_fd_num
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
        mut file: impl io::Read,
        len: usize,
        ptr: Pointer,
    ) -> InterpResult<'tcx, Result<usize, IoError>> {
        let this = self.eval_context_mut();

        let mut bytes = vec![0; len];
        let result = file.read(&mut bytes);
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

    /// Write data to a host `Write` type, withthe bytes taken from machine memory.
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
