//! General management of file descriptors, and support for
//! standard file descriptors (stdin/stdout/stderr).

use std::io;
use std::io::ErrorKind;

use rand::RngExt;
use rustc_abi::{Align, Size};
use rustc_target::spec::Os;

use crate::shims::FileDescriptionRef;
use crate::shims::files::{DynFileDescriptionRef, FileDescription};
use crate::shims::sig::check_min_vararg_count;
use crate::shims::unix::socket::UnixSocketFileDescription;
use crate::shims::unix::*;
use crate::*;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FlockOp {
    SharedLock { nonblocking: bool },
    ExclusiveLock { nonblocking: bool },
    Unlock,
}

/// Represents unix-specific file descriptions.
pub trait UnixFileDescription: FileDescription {
    /// Reads as much as possible into the given buffer `ptr` from a given offset.
    /// `len` indicates how many bytes we should try to read.
    /// `dest` is where the return value should be stored: number of bytes read, or `-1` in case of error.
    fn pread<'tcx>(
        &self,
        _communicate_allowed: bool,
        _offset: u64,
        _ptr: Pointer,
        _len: usize,
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
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
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot pwrite to {}", self.name());
    }

    fn flock<'tcx>(
        &self,
        _communicate_allowed: bool,
        _op: FlockOp,
    ) -> InterpResult<'tcx, io::Result<()>> {
        throw_unsup_format!("cannot flock {}", self.name());
    }

    /// Modifies device parameters.
    /// `op` is the device-dependent operation code. It's either a `c_long` or `c_int`, depending on
    /// the target and whether it uses glibc or musl.
    /// `arg` is the optional third argument which exists depending on the operation code. It's either
    /// an integer or a pointer.
    fn ioctl<'tcx>(
        &self,
        _op: Scalar,
        _arg: Option<&OpTy<'tcx>>,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, i32> {
        throw_unsup_format!("cannot use ioctl on {}", self.name());
    }

    /// Returns this file description as a Unix socket, if it represents one.
    fn as_socket<'tcx>(
        self: FileDescriptionRef<Self>,
        _ecx: &MiriInterpCx<'tcx>,
    ) -> Option<FileDescriptionRef<dyn UnixSocketFileDescription>> {
        None
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn dup(&mut self, old_fd_num: i32) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let Some(fd) = this.machine.fds.get(old_fd_num) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        interp_ok(Scalar::from_i32(this.machine.fds.insert(fd)))
    }

    fn dup2(&mut self, old_fd_num: i32, new_fd_num: i32) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let Some(fd) = this.machine.fds.get(old_fd_num) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        if new_fd_num != old_fd_num {
            // Close new_fd if it is previously opened.
            // If old_fd and new_fd point to the same description, then `dup_fd` ensures we keep the underlying file description alive.
            if let Some(old_new_fd) = this.machine.fds.fds.insert(new_fd_num, fd) {
                drop(old_new_fd);
            }
        }
        interp_ok(Scalar::from_i32(new_fd_num))
    }

    fn flock(&mut self, fd_num: i32, op: i32) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        let Some(fd) = this.machine.fds.get(fd_num) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
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

        let result = fd.as_unix(this).flock(this.machine.communicate(), parsed_op)?;
        // return `0` if flock is successful
        let result = result.map(|()| 0i32);
        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    fn ioctl(
        &mut self,
        fd: &OpTy<'tcx>,
        op: &OpTy<'tcx>,
        varargs: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd)?.to_i32()?;
        let op = this.read_scalar(op)?;
        // There is at most one relevant variadic argument.
        // It exists depending on the device and the opcode and thus we can't
        // use `check_min_vararg_count` here.
        let arg = varargs.first();

        let Some(fd) = this.machine.fds.get(fd) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };

        // Handle common opcodes.
        let fioclex = this.eval_libc("FIOCLEX");
        let fionclex = this.eval_libc("FIONCLEX");
        if op == fioclex || op == fionclex {
            // Since we don't support `exec`, those are NOPs.
            return interp_ok(Scalar::from_i32(0));
        }

        // Since some ioctl operations use the return value as an output parameter, we cannot strictly use the convention of
        // zero indicating success and -1 indicating an error.
        let return_value = fd.as_unix(this).ioctl(op, arg, this)?;
        interp_ok(Scalar::from_i32(return_value))
    }

    fn fcntl(
        &mut self,
        fd_num: &OpTy<'tcx>,
        cmd: &OpTy<'tcx>,
        varargs: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let fd_num = this.read_scalar(fd_num)?.to_i32()?;
        let cmd = this.read_scalar(cmd)?.to_i32()?;

        let f_getfd = this.eval_libc_i32("F_GETFD");
        let f_dupfd = this.eval_libc_i32("F_DUPFD");
        let f_dupfd_cloexec = this.eval_libc_i32("F_DUPFD_CLOEXEC");
        let f_getfl = this.eval_libc_i32("F_GETFL");
        let f_setfl = this.eval_libc_i32("F_SETFL");

        // We only support getting the flags for a descriptor.
        match cmd {
            cmd if cmd == f_getfd => {
                // Currently this is the only flag that `F_GETFD` returns. It is OK to just return the
                // `FD_CLOEXEC` value without checking if the flag is set for the file because `std`
                // always sets this flag when opening a file. However we still need to check that the
                // file itself is open.
                if !this.machine.fds.is_fd_num(fd_num) {
                    this.set_errno_and_return_neg1_i32(LibcError("EBADF"))
                } else {
                    interp_ok(this.eval_libc("FD_CLOEXEC"))
                }
            }
            cmd if cmd == f_dupfd || cmd == f_dupfd_cloexec => {
                // Note that we always assume the FD_CLOEXEC flag is set for every open file, in part
                // because exec() isn't supported. The F_DUPFD and F_DUPFD_CLOEXEC commands only
                // differ in whether the FD_CLOEXEC flag is pre-set on the new file descriptor,
                // thus they can share the same implementation here.
                let cmd_name = if cmd == f_dupfd {
                    "fcntl(fd, F_DUPFD, ...)"
                } else {
                    "fcntl(fd, F_DUPFD_CLOEXEC, ...)"
                };

                let [start] = check_min_vararg_count(cmd_name, varargs)?;
                let start = this.read_scalar(start)?.to_i32()?;

                if let Some(fd) = this.machine.fds.get(fd_num) {
                    interp_ok(Scalar::from_i32(this.machine.fds.insert_with_min_num(fd, start)))
                } else {
                    this.set_errno_and_return_neg1_i32(LibcError("EBADF"))
                }
            }
            cmd if cmd == f_getfl => {
                // Check if this is a valid open file descriptor.
                let Some(fd) = this.machine.fds.get(fd_num) else {
                    return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
                };

                fd.get_flags(this)
            }
            cmd if cmd == f_setfl => {
                // Check if this is a valid open file descriptor.
                let Some(fd) = this.machine.fds.get(fd_num) else {
                    return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
                };

                let [flag] = check_min_vararg_count("fcntl(fd, F_SETFL, ...)", varargs)?;
                let flag = this.read_scalar(flag)?.to_i32()?;

                // Ignore flags that never get stored by SETFL.
                // "File access mode (O_RDONLY, O_WRONLY, O_RDWR) and file
                // creation flags (i.e., O_CREAT, O_EXCL, O_NOCTTY, O_TRUNC)
                // in arg are ignored."
                let ignored_flags = this.eval_libc_i32("O_RDONLY")
                    | this.eval_libc_i32("O_WRONLY")
                    | this.eval_libc_i32("O_RDWR")
                    | this.eval_libc_i32("O_CREAT")
                    | this.eval_libc_i32("O_EXCL")
                    | this.eval_libc_i32("O_NOCTTY")
                    | this.eval_libc_i32("O_TRUNC");

                fd.set_flags(flag & !ignored_flags, this)
            }
            cmd if this.tcx.sess.target.os == Os::MacOs
                && cmd == this.eval_libc_i32("F_FULLFSYNC") =>
            {
                // Reject if isolation is enabled.
                if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
                    this.reject_in_isolation("`fcntl`", reject_with)?;
                    return this.set_errno_and_return_neg1_i32(ErrorKind::PermissionDenied);
                }

                this.ffullsync_fd(fd_num)
            }
            cmd => {
                throw_unsup_format!("fcntl: unsupported command {cmd:#x}");
            }
        }
    }

    fn close(&mut self, fd_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let fd_num = this.read_scalar(fd_op)?.to_i32()?;

        let Some(fd) = this.machine.fds.remove(fd_num) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        drop(fd);
        // Our close is always successful. Close does not reliably return errors anyway so it is
        // not worth the effort to try and return anything here.
        interp_ok(Scalar::from_i32(0))
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
        this.check_ptr_access(buf, Size::from_bytes(count), CheckInAllocMsg::MemoryAccess)?;

        // We cap the number of read bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count
            .min(u64::try_from(this.target_isize_max()).unwrap())
            .min(u64::try_from(isize::MAX).unwrap());
        let count = usize::try_from(count).unwrap(); // now it fits in a `usize`

        // Get the FD.
        let Some(fd) = this.machine.fds.get(fd_num) else {
            trace!("read: FD not found");
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        trace!("read: FD mapped to {fd:?}");
        // We want to read at most `count` bytes. We are sure that `count` is not negative
        // because it was a target's `usize`. Also we are sure that it's smaller than
        // `usize::MAX` because it is bounded by the host's `isize`.

        let dest = dest.clone();
        this.read_from_fd(
            fd,
            buf,
            count,
            offset,
            callback!(
                @capture<'tcx> {
                    count: usize,
                    dest: MPlaceTy<'tcx>,
                }
                |this, result: Result<usize, IoError>| {
                    match result {
                        Ok(read_size) => {
                            assert!(read_size <= count);
                            // This must fit since `count` fits.
                            this.write_int(u64::try_from(read_size).unwrap(), &dest)
                        }
                        Err(e) => this.set_errno_and_return_neg1(e, &dest)
                }}
            ),
        )
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
        this.check_ptr_access(buf, Size::from_bytes(count), CheckInAllocMsg::MemoryAccess)?;

        // We cap the number of written bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count
            .min(u64::try_from(this.target_isize_max()).unwrap())
            .min(u64::try_from(isize::MAX).unwrap());
        let count = usize::try_from(count).unwrap(); // now it fits in a `usize`

        // We temporarily dup the FD to be able to retain mutable access to `this`.
        let Some(fd) = this.machine.fds.get(fd_num) else {
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        let dest = dest.clone();
        this.write_to_fd(
            fd,
            buf,
            count,
            offset,
            callback!(
                @capture<'tcx> {
                    count: usize,
                    dest: MPlaceTy<'tcx>,
                }
                |this, result: Result<usize, IoError>| {
                    match result {
                        Ok(write_size) => {
                            assert!(write_size <= count);
                            // This must fit since `count` fits.
                            this.write_int(u64::try_from(write_size).unwrap(), &dest)
                        }
                        Err(e) => this.set_errno_and_return_neg1(e, &dest)

                }}
            ),
        )
    }

    /// Vectored reads are implemented by first reading bytes from `fd`
    /// into a temporary buffer which has the combined size of all buffers in
    /// `iov`. After that we split the bytes of the combined buffer into the
    /// buffers of `iov`. This ensures that the vectored read occurs atomically.
    fn readv(
        &mut self,
        fd: &OpTy<'tcx>,
        iov: &OpTy<'tcx>,
        iovcnt: &OpTy<'tcx>,
        offset: Option<&OpTy<'tcx>>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd)?.to_i32()?;
        let iov_ptr = this.read_pointer(iov)?;
        let iovcnt: u64 = this.read_scalar(iovcnt)?.to_i32()?.try_into().unwrap();
        // `readv` is the same as `preadv` without an offset.
        let offset = if let Some(offset) = offset {
            if matches!(this.tcx.sess.target.os, Os::Solaris) {
                throw_unsup_format!(
                    "preadv: vectored reads with offsets aren't supported on Solaris"
                )
            }
            Some(this.read_scalar(offset)?.to_int(offset.layout.size)?)
        } else {
            None
        };

        // Check that the FD exists.
        let Some(fd) = this.machine.fds.get(fd) else {
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        let iovec_layout = this.libc_array_ty_layout("iovec", iovcnt);
        let iov_ptr_mplace = this.ptr_to_mplace(iov_ptr, iovec_layout);

        // Read list of buffers from `iov`.
        let mut buffers = Vec::new();

        let mut array = this.project_array_fields(&iov_ptr_mplace)?;
        while let Some((_idx, iovec)) = array.next(this)? {
            let iov_len_field = this.project_field_named(&iovec, "iov_len")?;
            let iov_len: u64 = this
                .read_scalar(&iov_len_field)?
                .to_int(iov_len_field.layout.size)?
                .try_into()
                .unwrap();

            let iov_base_field = this.project_field_named(&iovec, "iov_base")?;
            let iov_base_ptr = this.read_pointer(&iov_base_field)?;

            buffers.push((iov_base_ptr, iov_len));
        }

        let total_bytes = buffers.iter().map(|(_, len)| len).sum::<u64>();

        // Allocate a temporary buffer which has the combined size of all buffers provided in `iov`.
        let tmp_ptr: Pointer = this
            .allocate_ptr(
                Size::from_bytes(total_bytes),
                Align::ONE,
                MemoryKind::Stack,
                AllocInit::Uninit,
            )?
            .into();

        let dest = dest.clone();
        this.read_from_fd(
            fd,
            tmp_ptr,
            usize::try_from(total_bytes).unwrap(),
            offset,
            callback!(
                @capture<'tcx> {
                    tmp_ptr: Pointer,
                    buffers: Vec<(Pointer, u64)>,
                    dest: MPlaceTy<'tcx>
                } |this, result: Result<usize, IoError>| {
                    let bytes_read = match result {
                        Ok(size) => {
                            this.write_scalar(Scalar::from_target_isize(size.try_into().unwrap(), this), &dest)?;
                            u64::try_from(size).unwrap()
                        },
                        Err(e) => {
                            this.deallocate_ptr(tmp_ptr, None, MemoryKind::Stack)?;
                            return this.set_errno_and_return_neg1(e, &dest)
                        }
                    };
                    let mut remaining_bytes = bytes_read;

                    // Split the bytes from the temporary buffer into the buffers provided in `iov`.
                    // We start at the first buffer and fill them in order, until we reach the end of the
                    // initialized bytes in the temporary buffer.
                    for (buffer_ptr, buffer_len) in buffers {
                        // Offset temporary buffer by the amount of bytes we already copied into previous buffers.
                        let tmp_ptr_with_offset =
                            this.ptr_offset_inbounds(tmp_ptr, i64::try_from(bytes_read.strict_sub(remaining_bytes)).unwrap())?;

                        // Copy at most as many bytes as the buffer fits but without reading
                        // any uninitialized bytes from the temporary buffer.
                        let copy_amount = buffer_len.min(remaining_bytes);
                        this.mem_copy(
                            tmp_ptr_with_offset,
                            buffer_ptr,
                            Size::from_bytes(copy_amount),
                            // The buffers are guaranteed to not overlap because we just newly allocated
                            // the `tmp_ptr`, and `tmp_ptr_with_offset` is guaranteed to be
                            // within those boundaries.
                            true,
                        )?;

                        remaining_bytes = remaining_bytes.strict_sub(copy_amount);
                        if remaining_bytes == 0 {
                            // We don't have anything left to copy; exit the loop.
                            break;
                        }
                    }

                    this.deallocate_ptr(tmp_ptr, None, MemoryKind::Stack)
                }),
        )
    }

    /// Vectored writes are implemented by first writing the bytes from all
    /// buffers of `iov` into a combined temporary buffer and then writing this
    /// combined buffer into `fd`. This ensures that the vectored write occurs atomically.
    fn writev(
        &mut self,
        fd: &OpTy<'tcx>,
        iov: &OpTy<'tcx>,
        iovcnt: &OpTy<'tcx>,
        offset: Option<&OpTy<'tcx>>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd)?.to_i32()?;
        let iov_ptr = this.read_pointer(iov)?;
        let iovcnt: u64 = this.read_scalar(iovcnt)?.to_i32()?.try_into().unwrap();
        // `writev` is the same as `pwritev` without an offset.
        let offset = if let Some(offset) = offset {
            if matches!(this.tcx.sess.target.os, Os::Solaris) {
                throw_unsup_format!(
                    "pwritev: vectored writes with offsets aren't supported on Solaris"
                )
            }
            Some(this.read_scalar(offset)?.to_int(offset.layout.size)?)
        } else {
            None
        };

        // Check that the FD exists.
        let Some(fd) = this.machine.fds.get(fd) else {
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        let iovec_layout = this.libc_array_ty_layout("iovec", iovcnt);
        let iov_ptr_mplace = this.ptr_to_mplace(iov_ptr, iovec_layout);

        // Read list of buffers from `iov`.
        let mut buffers = Vec::new();

        let mut array = this.project_array_fields(&iov_ptr_mplace)?;
        while let Some((_idx, iovec)) = array.next(this)? {
            let iov_len_field = this.project_field_named(&iovec, "iov_len")?;
            let iov_len: u64 = this
                .read_scalar(&iov_len_field)?
                .to_int(iov_len_field.layout.size)?
                .try_into()
                .unwrap();

            let iov_base_field = this.project_field_named(&iovec, "iov_base")?;
            let iov_base_ptr = this.read_pointer(&iov_base_field)?;

            buffers.push((iov_base_ptr, iov_len));
        }

        let total_bytes = buffers.iter().map(|(_, len)| len).sum::<u64>();

        // Allocate a temporary buffer which has the combined size of all buffers provided in `iov`.
        let tmp_ptr: Pointer = this
            .allocate_ptr(
                Size::from_bytes(total_bytes),
                Align::ONE,
                MemoryKind::Stack,
                AllocInit::Uninit,
            )?
            .into();

        // Copy the bytes from all buffers provided in `iov` into the temporary buffer.
        // We start at the first buffer and then continue buffer by buffer.
        let mut bytes_copied: u64 = 0;
        for (buffer_ptr, buffer_len) in buffers {
            // Offset temporary buffer by the amount of bytes we already copied from previous buffers.
            let tmp_ptr_with_offset =
                this.ptr_offset_inbounds(tmp_ptr, i64::try_from(bytes_copied).unwrap())?;

            this.mem_copy(
                buffer_ptr,
                tmp_ptr_with_offset,
                Size::from_bytes(buffer_len),
                // The buffers are guaranteed to not overlap because we just newly allocated
                // the `tmp_ptr`, and `tmp_ptr_with_offset` is guaranteed to be
                // within those boundaries.
                true,
            )?;

            bytes_copied = bytes_copied.strict_add(buffer_len);
        }

        let dest = dest.clone();
        // Write bytes from the temporary buffer. This ensures the write is atomic.
        this.write_to_fd(
            fd,
            tmp_ptr,
            usize::try_from(total_bytes).unwrap(),
            offset,
            callback!(
                @capture<'tcx> {
                    tmp_ptr: Pointer,
                    dest: MPlaceTy<'tcx>,
                }
                |this, result: Result<usize, IoError>| {
                    this.deallocate_ptr(tmp_ptr, None, MemoryKind::Stack)?;
                    match result {
                        Ok(size) => this.write_scalar(Scalar::from_target_isize(size.try_into().unwrap(), this), &dest),
                        Err(e) => this.set_errno_and_return_neg1(e, &dest)
                    }
            }),
        )
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Read `len` bytes from the `fd` file description at `offset` into the buffer
    /// pointed to by `ptr`.
    /// If `offset` is [`Some`], the read occurs at the given absolute position rather
    /// than the current file position (`read_at` semantics rather than `read`).
    /// `finish` will be invoked when the read is done (which might be way after
    /// this function returns as the read may block).
    fn read_from_fd(
        &mut self,
        fd: DynFileDescriptionRef,
        ptr: Pointer,
        len: usize,
        offset: Option<i128>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Handle the zero-sized case. The man page says:
        // > If count is zero, read() may detect the errors described below.  In the absence of any
        // > errors, or if read() does not check for errors, a read() with a count of 0 returns zero
        // > and has no other effects.
        if len == 0 {
            return finish.call(this, Ok(0));
        }

        // Non-deterministically decide to further reduce the length, simulating a partial read (but
        // never to 0, that would indicate EOF).
        let len = if this.machine.short_fd_operations
            && fd.short_fd_operations()
            && len >= 2
            && this.machine.rng.get_mut().random()
        {
            len / 2 // since `len` is at least 2, the result is still at least 1
        } else {
            len
        };

        match offset {
            None => fd.read(this.machine.communicate(), ptr, len, this, finish)?,
            Some(offset) => {
                let Ok(offset) = u64::try_from(offset) else {
                    return finish.call(this, Err(LibcError("EINVAL")));
                };
                fd.as_unix(this).pread(
                    this.machine.communicate(),
                    offset,
                    ptr,
                    len,
                    this,
                    finish,
                )?
            }
        };
        interp_ok(())
    }

    /// Write `len` bytes at `offset` from the buffer pointed to by `ptr` into the `fd`
    /// file description.
    /// If `offset` is [`Some`], the write occurs at the given absolute position rather
    /// than the current file position (`write_at` semantics rather than `write`).
    /// `finish` will be invoked when the write is done (which might be way after
    /// this function returns as the write may block).
    fn write_to_fd(
        &mut self,
        fd: DynFileDescriptionRef,
        ptr: Pointer,
        len: usize,
        offset: Option<i128>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Handle the zero-sized case. The man page says:
        // > If count is zero and fd refers to a regular file, then write() may return a failure
        // > status if one of the errors below is detected.  If no errors are detected, or error
        // > detection is not performed, 0 is returned without causing any other effect.   If  count
        // > is  zero  and  fd refers to a file other than a regular file, the results are not
        // > specified.
        if len == 0 {
            // For now let's not open the can of worms of what exactly "not specified" could mean...
            return finish.call(this, Ok(0));
        }

        // Non-deterministically decide to further reduce the length, simulating a partial write.
        // We avoid reducing the write size to 0: the docs seem to be entirely fine with that,
        // but the standard library is not (https://github.com/rust-lang/rust/issues/145959).
        let len = if this.machine.short_fd_operations
            && fd.short_fd_operations()
            && len >= 2
            && this.machine.rng.get_mut().random()
        {
            len / 2
        } else {
            len
        };

        match offset {
            None => fd.write(this.machine.communicate(), ptr, len, this, finish)?,
            Some(offset) => {
                let Ok(offset) = u64::try_from(offset) else {
                    return finish.call(this, Err(LibcError("EINVAL")));
                };
                fd.as_unix(this).pwrite(
                    this.machine.communicate(),
                    ptr,
                    len,
                    offset,
                    this,
                    finish,
                )?
            }
        };
        interp_ok(())
    }
}
