//! General management of file descriptors, and support for
//! standard file descriptors (stdin/stdout/stderr).

use std::any::Any;
use std::collections::BTreeMap;
use std::io::{self, ErrorKind, IsTerminal, Read, SeekFrom, Write};
use std::ops::Deref;
use std::rc::Rc;
use std::rc::Weak;

use rustc_target::abi::Size;

use crate::shims::unix::linux::epoll::EpollReadyEvents;
use crate::shims::unix::*;
use crate::*;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum FlockOp {
    SharedLock { nonblocking: bool },
    ExclusiveLock { nonblocking: bool },
    Unlock,
}

/// Represents an open file description.
pub trait FileDescription: std::fmt::Debug + Any {
    fn name(&self) -> &'static str;

    /// Reads as much as possible into the given buffer `ptr`.
    /// `len` indicates how many bytes we should try to read.
    /// `dest` is where the return value should be stored: number of bytes read, or `-1` in case of error.
    fn read<'tcx>(
        &self,
        _self_ref: &FileDescriptionRef,
        _communicate_allowed: bool,
        _ptr: Pointer,
        _len: usize,
        _dest: &MPlaceTy<'tcx>,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot read from {}", self.name());
    }

    /// Writes as much as possible from the given buffer `ptr`.
    /// `len` indicates how many bytes we should try to write.
    /// `dest` is where the return value should be stored: number of bytes written, or `-1` in case of error.
    fn write<'tcx>(
        &self,
        _self_ref: &FileDescriptionRef,
        _communicate_allowed: bool,
        _ptr: Pointer,
        _len: usize,
        _dest: &MPlaceTy<'tcx>,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot write to {}", self.name());
    }

    /// Reads as much as possible into the given buffer `ptr` from a given offset.
    /// `len` indicates how many bytes we should try to read.
    /// `dest` is where the return value should be stored: number of bytes read, or `-1` in case of error.
    fn pread<'tcx>(
        &self,
        _communicate_allowed: bool,
        _offset: u64,
        _ptr: Pointer,
        _len: usize,
        _dest: &MPlaceTy<'tcx>,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot pread from {}", self.name());
    }

    /// Writes as much as possible from the given buffer `ptr` starting at a given offset.
    /// `ptr` is the pointer to the user supplied read buffer.
    /// `len` indicates how many bytes we should try to write.
    /// `dest` is where the return value should be stored: number of bytes written, or `-1` in case of error.
    fn pwrite<'tcx>(
        &self,
        _communicate_allowed: bool,
        _ptr: Pointer,
        _len: usize,
        _offset: u64,
        _dest: &MPlaceTy<'tcx>,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot pwrite to {}", self.name());
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

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        throw_unsup_format!("cannot close {}", self.name());
    }

    fn flock<'tcx>(
        &self,
        _communicate_allowed: bool,
        _op: FlockOp,
    ) -> InterpResult<'tcx, io::Result<()>> {
        throw_unsup_format!("cannot flock {}", self.name());
    }

    fn is_tty(&self, _communicate_allowed: bool) -> bool {
        // Most FDs are not tty's and the consequence of a wrong `false` are minor,
        // so we use a default impl here.
        false
    }

    /// Check the readiness of file description.
    fn get_epoll_ready_events<'tcx>(&self) -> InterpResult<'tcx, EpollReadyEvents> {
        throw_unsup_format!("{}: epoll does not support this file description", self.name());
    }
}

impl dyn FileDescription {
    #[inline(always)]
    pub fn downcast<T: Any>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref()
    }
}

impl FileDescription for io::Stdin {
    fn name(&self) -> &'static str {
        "stdin"
    }

    fn read<'tcx>(
        &self,
        _self_ref: &FileDescriptionRef,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        let mut bytes = vec![0; len];
        if !communicate_allowed {
            // We want isolation mode to be deterministic, so we have to disallow all reads, even stdin.
            helpers::isolation_abort_error("`read` from stdin")?;
        }
        let result = Read::read(&mut { self }, &mut bytes);
        ecx.return_read_bytes_and_count(ptr, &bytes, result, dest)
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
        &self,
        _self_ref: &FileDescriptionRef,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        let bytes = ecx.read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(len))?;
        // We allow writing to stderr even with isolation enabled.
        let result = Write::write(&mut { self }, bytes);
        // Stdout is buffered, flush to make sure it appears on the
        // screen.  This is the write() syscall of the interpreted
        // program, we want it to correspond to a write() syscall on
        // the host -- there is no good in adding extra buffering
        // here.
        io::stdout().flush().unwrap();
        ecx.return_written_byte_count_or_error(result, dest)
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
        &self,
        _self_ref: &FileDescriptionRef,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        let bytes = ecx.read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(len))?;
        // We allow writing to stderr even with isolation enabled.
        // No need to flush, stderr is not buffered.
        let result = Write::write(&mut { self }, bytes);
        ecx.return_written_byte_count_or_error(result, dest)
    }

    fn is_tty(&self, communicate_allowed: bool) -> bool {
        communicate_allowed && self.is_terminal()
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
        &self,
        _self_ref: &FileDescriptionRef,
        _communicate_allowed: bool,
        _ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        // We just don't write anything, but report to the user that we did.
        let result = Ok(len);
        ecx.return_written_byte_count_or_error(result, dest)
    }
}

/// Structure contains both the file description and its unique identifier.
#[derive(Clone, Debug)]
pub struct FileDescWithId<T: FileDescription + ?Sized> {
    id: FdId,
    file_description: Box<T>,
}

#[derive(Clone, Debug)]
pub struct FileDescriptionRef(Rc<FileDescWithId<dyn FileDescription>>);

impl Deref for FileDescriptionRef {
    type Target = dyn FileDescription;

    fn deref(&self) -> &Self::Target {
        &*self.0.file_description
    }
}

impl FileDescriptionRef {
    fn new(fd: impl FileDescription, id: FdId) -> Self {
        FileDescriptionRef(Rc::new(FileDescWithId { id, file_description: Box::new(fd) }))
    }

    pub fn close<'tcx>(
        self,
        communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        // Destroy this `Rc` using `into_inner` so we can call `close` instead of
        // implicitly running the destructor of the file description.
        let id = self.get_id();
        match Rc::into_inner(self.0) {
            Some(fd) => {
                // Remove entry from the global epoll_event_interest table.
                ecx.machine.epoll_interests.remove(id);

                fd.file_description.close(communicate_allowed, ecx)
            }
            None => Ok(Ok(())),
        }
    }

    pub fn downgrade(&self) -> WeakFileDescriptionRef {
        WeakFileDescriptionRef { weak_ref: Rc::downgrade(&self.0) }
    }

    pub fn get_id(&self) -> FdId {
        self.0.id
    }
}

/// Holds a weak reference to the actual file description.
#[derive(Clone, Debug, Default)]
pub struct WeakFileDescriptionRef {
    weak_ref: Weak<FileDescWithId<dyn FileDescription>>,
}

impl WeakFileDescriptionRef {
    pub fn upgrade(&self) -> Option<FileDescriptionRef> {
        if let Some(file_desc_with_id) = self.weak_ref.upgrade() {
            return Some(FileDescriptionRef(file_desc_with_id));
        }
        None
    }
}

impl VisitProvenance for WeakFileDescriptionRef {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // A weak reference can never be the only reference to some pointer or place.
        // Since the actual file description is tracked by strong ref somewhere,
        // it is ok to make this a NOP operation.
    }
}

/// A unique id for file descriptions. While we could use the address, considering that
/// is definitely unique, the address would expose interpreter internal state when used
/// for sorting things. So instead we generate a unique id per file description that stays
/// the same even if a file descriptor is duplicated and gets a new integer file descriptor.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct FdId(usize);

/// The file descriptor table
#[derive(Debug)]
pub struct FdTable {
    pub fds: BTreeMap<i32, FileDescriptionRef>,
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

    pub fn new_ref(&mut self, fd: impl FileDescription) -> FileDescriptionRef {
        let file_handle = FileDescriptionRef::new(fd, self.next_file_description_id);
        self.next_file_description_id = FdId(self.next_file_description_id.0.strict_add(1));
        file_handle
    }

    /// Insert a new file description to the FdTable.
    pub fn insert_new(&mut self, fd: impl FileDescription) -> i32 {
        let fd_ref = self.new_ref(fd);
        self.insert(fd_ref)
    }

    pub fn insert(&mut self, fd_ref: FileDescriptionRef) -> i32 {
        self.insert_with_min_num(fd_ref, 0)
    }

    /// Insert a file description, giving it a file descriptor that is at least `min_fd_num`.
    fn insert_with_min_num(&mut self, file_handle: FileDescriptionRef, min_fd_num: i32) -> i32 {
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

    pub fn get(&self, fd_num: i32) -> Option<FileDescriptionRef> {
        let fd = self.fds.get(&fd_num)?;
        Some(fd.clone())
    }

    pub fn remove(&mut self, fd_num: i32) -> Option<FileDescriptionRef> {
        self.fds.remove(&fd_num)
    }

    pub fn is_fd_num(&self, fd_num: i32) -> bool {
        self.fds.contains_key(&fd_num)
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn dup(&mut self, old_fd_num: i32) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let Some(fd) = this.machine.fds.get(old_fd_num) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };
        Ok(Scalar::from_i32(this.machine.fds.insert(fd)))
    }

    fn dup2(&mut self, old_fd_num: i32, new_fd_num: i32) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let Some(fd) = this.machine.fds.get(old_fd_num) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };
        if new_fd_num != old_fd_num {
            // Close new_fd if it is previously opened.
            // If old_fd and new_fd point to the same description, then `dup_fd` ensures we keep the underlying file description alive.
            if let Some(old_new_fd) = this.machine.fds.fds.insert(new_fd_num, fd) {
                // Ignore close error (not interpreter's) according to dup2() doc.
                old_new_fd.close(this.machine.communicate(), this)?.ok();
            }
        }
        Ok(Scalar::from_i32(new_fd_num))
    }

    fn flock(&mut self, fd_num: i32, op: i32) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        let Some(fd) = this.machine.fds.get(fd_num) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };

        // We need to check that there aren't unsupported options in `op`.
        let lock_sh = this.eval_libc_i32("LOCK_SH");
        let lock_ex = this.eval_libc_i32("LOCK_EX");
        let lock_nb = this.eval_libc_i32("LOCK_NB");
        let lock_un = this.eval_libc_i32("LOCK_UN");

        use FlockOp::*;
        let parsed_op = if op == lock_sh {
            SharedLock { nonblocking: false }
        } else if op == lock_sh | lock_nb {
            SharedLock { nonblocking: true }
        } else if op == lock_ex {
            ExclusiveLock { nonblocking: false }
        } else if op == lock_ex | lock_nb {
            ExclusiveLock { nonblocking: true }
        } else if op == lock_un {
            Unlock
        } else {
            throw_unsup_format!("unsupported flags {:#x}", op);
        };

        let result = fd.flock(this.machine.communicate(), parsed_op)?;
        drop(fd);
        // return `0` if flock is successful
        let result = result.map(|()| 0i32);
        Ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    fn fcntl(&mut self, args: &[OpTy<'tcx>]) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        if args.len() < 2 {
            throw_ub_format!(
                "incorrect number of arguments for fcntl: got {}, expected at least 2",
                args.len()
            );
        }
        let fd_num = this.read_scalar(&args[0])?.to_i32()?;
        let cmd = this.read_scalar(&args[1])?.to_i32()?;

        // We only support getting the flags for a descriptor.
        if cmd == this.eval_libc_i32("F_GETFD") {
            // Currently this is the only flag that `F_GETFD` returns. It is OK to just return the
            // `FD_CLOEXEC` value without checking if the flag is set for the file because `std`
            // always sets this flag when opening a file. However we still need to check that the
            // file itself is open.
            Ok(Scalar::from_i32(if this.machine.fds.is_fd_num(fd_num) {
                this.eval_libc_i32("FD_CLOEXEC")
            } else {
                this.fd_not_found()?
            }))
        } else if cmd == this.eval_libc_i32("F_DUPFD")
            || cmd == this.eval_libc_i32("F_DUPFD_CLOEXEC")
        {
            // Note that we always assume the FD_CLOEXEC flag is set for every open file, in part
            // because exec() isn't supported. The F_DUPFD and F_DUPFD_CLOEXEC commands only
            // differ in whether the FD_CLOEXEC flag is pre-set on the new file descriptor,
            // thus they can share the same implementation here.
            if args.len() < 3 {
                throw_ub_format!(
                    "incorrect number of arguments for fcntl with cmd=`F_DUPFD`/`F_DUPFD_CLOEXEC`: got {}, expected at least 3",
                    args.len()
                );
            }
            let start = this.read_scalar(&args[2])?.to_i32()?;

            match this.machine.fds.get(fd_num) {
                Some(fd) => Ok(Scalar::from_i32(this.machine.fds.insert_with_min_num(fd, start))),
                None => Ok(Scalar::from_i32(this.fd_not_found()?)),
            }
        } else if this.tcx.sess.target.os == "macos" && cmd == this.eval_libc_i32("F_FULLFSYNC") {
            // Reject if isolation is enabled.
            if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
                this.reject_in_isolation("`fcntl`", reject_with)?;
                this.set_last_error_from_io_error(ErrorKind::PermissionDenied.into())?;
                return Ok(Scalar::from_i32(-1));
            }

            this.ffullsync_fd(fd_num)
        } else {
            throw_unsup_format!("the {:#x} command is not supported for `fcntl`)", cmd);
        }
    }

    fn close(&mut self, fd_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let fd_num = this.read_scalar(fd_op)?.to_i32()?;

        let Some(fd) = this.machine.fds.remove(fd_num) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };
        let result = fd.close(this.machine.communicate(), this)?;
        // return `0` if close is successful
        let result = result.map(|()| 0i32);
        Ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    /// Function used when a file descriptor does not exist. It returns `Ok(-1)`and sets
    /// the last OS error to `libc::EBADF` (invalid file descriptor). This function uses
    /// `T: From<i32>` instead of `i32` directly because some fs functions return different integer
    /// types (like `read`, that returns an `i64`).
    fn fd_not_found<T: From<i32>>(&mut self) -> InterpResult<'tcx, T> {
        let this = self.eval_context_mut();
        let ebadf = this.eval_libc("EBADF");
        this.set_last_error(ebadf)?;
        Ok((-1).into())
    }

    /// Read data from `fd` into buffer specified by `buf` and `count`.
    ///
    /// If `offset` is `None`, reads data from current cursor position associated with `fd`
    /// and updates cursor position on completion. Otherwise, reads from the specified offset
    /// and keeps the cursor unchanged.
    fn read(
        &mut self,
        fd_num: i32,
        buf: Pointer,
        count: u64,
        offset: Option<i128>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Isolation check is done via `FileDescription` trait.

        trace!("Reading from FD {}, size {}", fd_num, count);

        // Check that the *entire* buffer is actually valid memory.
        this.check_ptr_access(buf, Size::from_bytes(count), CheckInAllocMsg::MemoryAccessTest)?;

        // We cap the number of read bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count
            .min(u64::try_from(this.target_isize_max()).unwrap())
            .min(u64::try_from(isize::MAX).unwrap());
        let count = usize::try_from(count).unwrap(); // now it fits in a `usize`
        let communicate = this.machine.communicate();

        // We temporarily dup the FD to be able to retain mutable access to `this`.
        let Some(fd) = this.machine.fds.get(fd_num) else {
            trace!("read: FD not found");
            let res: i32 = this.fd_not_found()?;
            this.write_int(res, dest)?;
            return Ok(());
        };

        trace!("read: FD mapped to {fd:?}");
        // We want to read at most `count` bytes. We are sure that `count` is not negative
        // because it was a target's `usize`. Also we are sure that its smaller than
        // `usize::MAX` because it is bounded by the host's `isize`.

        match offset {
            None => fd.read(&fd, communicate, buf, count, dest, this)?,
            Some(offset) => {
                let Ok(offset) = u64::try_from(offset) else {
                    let einval = this.eval_libc("EINVAL");
                    this.set_last_error(einval)?;
                    this.write_int(-1, dest)?;
                    return Ok(());
                };
                fd.pread(communicate, offset, buf, count, dest, this)?
            }
        };
        Ok(())
    }

    fn write(
        &mut self,
        fd_num: i32,
        buf: Pointer,
        count: u64,
        offset: Option<i128>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Isolation check is done via `FileDescription` trait.

        // Check that the *entire* buffer is actually valid memory.
        this.check_ptr_access(buf, Size::from_bytes(count), CheckInAllocMsg::MemoryAccessTest)?;

        // We cap the number of written bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count
            .min(u64::try_from(this.target_isize_max()).unwrap())
            .min(u64::try_from(isize::MAX).unwrap());
        let count = usize::try_from(count).unwrap(); // now it fits in a `usize`
        let communicate = this.machine.communicate();

        // We temporarily dup the FD to be able to retain mutable access to `this`.
        let Some(fd) = this.machine.fds.get(fd_num) else {
            let res: i32 = this.fd_not_found()?;
            this.write_int(res, dest)?;
            return Ok(());
        };

        match offset {
            None => fd.write(&fd, communicate, buf, count, dest, this)?,
            Some(offset) => {
                let Ok(offset) = u64::try_from(offset) else {
                    let einval = this.eval_libc("EINVAL");
                    this.set_last_error(einval)?;
                    this.write_int(-1, dest)?;
                    return Ok(());
                };
                fd.pwrite(communicate, buf, count, offset, dest, this)?
            }
        };
        Ok(())
    }

    /// Helper to implement `FileDescription::read`:
    /// `result` should be the return value of some underlying `read` call that used `bytes` as its output buffer.
    /// The length of `bytes` must not exceed either the host's or the target's `isize`.
    /// If `Result` indicates success, `bytes` is written to `buf` and the size is written to `dest`.
    /// Otherwise, `-1` is written to `dest` and the last libc error is set appropriately.
    fn return_read_bytes_and_count(
        &mut self,
        buf: Pointer,
        bytes: &[u8],
        result: io::Result<usize>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        match result {
            Ok(read_bytes) => {
                // If reading to `bytes` did not fail, we write those bytes to the buffer.
                // Crucially, if fewer than `bytes.len()` bytes were read, only write
                // that much into the output buffer!
                this.write_bytes_ptr(buf, bytes[..read_bytes].iter().copied())?;
                // The actual read size is always less than what got originally requested so this cannot fail.
                this.write_int(u64::try_from(read_bytes).unwrap(), dest)?;
                return Ok(());
            }
            Err(e) => {
                this.set_last_error_from_io_error(e)?;
                this.write_int(-1, dest)?;
                return Ok(());
            }
        }
    }

    /// This function writes the number of written bytes (given in `result`) to `dest`, or sets the
    /// last libc error and writes -1 to dest.
    fn return_written_byte_count_or_error(
        &mut self,
        result: io::Result<usize>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let result = this.try_unwrap_io_result(result.map(|c| i64::try_from(c).unwrap()))?;
        this.write_int(result, dest)?;
        Ok(())
    }
}
