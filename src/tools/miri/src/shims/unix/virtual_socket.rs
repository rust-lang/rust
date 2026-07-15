//! This implements "virtual" sockets, that do not correspond to anything on the host system and
//! are entirely implemented inside Miri.
//! This is used to implement `socketpair` and `pipe`.

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::VecDeque;
use std::io::{self, ErrorKind, Read};

use rustc_target::spec::Os;

use crate::concurrency::VClock;
use crate::shims::files::{
    EvalContextExt as _, FdId, FileDescription, FileDescriptionRef, WeakFileDescriptionRef,
};
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::socket::UnixSocketFileDescription;
use crate::*;

/// The maximum capacity of the socketpair buffer in bytes.
/// This number is arbitrary as the value can always
/// be configured in the real system.
const MAX_SOCKETPAIR_BUFFER_CAPACITY: usize = 0x34000;

#[derive(Debug, PartialEq)]
enum VirtualSocketType {
    // Either end of the socketpair fd.
    Socketpair,
    // Read end of the pipe.
    PipeRead,
    // Write end of the pipe.
    PipeWrite,
}

/// One end of a pair of connected virtual sockets.
#[derive(Debug)]
struct VirtualSocket {
    /// The buffer we are reading from, or `None` if this is the writing end of a pipe.
    /// (In that case, the peer FD will be the reading end of that pipe.)
    readbuf: Option<RefCell<Buffer>>,
    /// The `VirtualSocket` file descriptor that is our "peer", and that holds the buffer we are
    /// writing to. This is a weak reference because the other side may be closed before us; all
    /// future writes will then trigger EPIPE.
    peer_fd: OnceCell<WeakFileDescriptionRef<VirtualSocket>>,
    /// Indicates whether the peer has lost data when the file description is closed.
    /// This flag is set to `true` if the peer's `readbuf` is non-empty at the time
    /// of closure.
    peer_lost_data: Cell<bool>,
    /// A list of thread ids blocked because the buffer was empty.
    /// Once another thread writes some bytes, these threads will be unblocked.
    blocked_read_tid: RefCell<Vec<ThreadId>>,
    /// A list of thread ids blocked because the buffer was full.
    /// Once another thread reads some bytes, these threads will be unblocked.
    blocked_write_tid: RefCell<Vec<ThreadId>>,
    /// Whether this fd is non-blocking or not.
    is_nonblock: Cell<bool>,
    // Differentiate between different virtual socket fd types.
    fd_type: VirtualSocketType,
}

#[derive(Debug)]
struct Buffer {
    buf: VecDeque<u8>,
    clock: VClock,
}

impl Buffer {
    fn new() -> Self {
        Buffer { buf: VecDeque::new(), clock: VClock::default() }
    }
}

impl VirtualSocket {
    fn peer_fd(&self) -> &WeakFileDescriptionRef<VirtualSocket> {
        self.peer_fd.get().unwrap()
    }
}

impl FileDescription for VirtualSocket {
    fn name(&self) -> &'static str {
        match self.fd_type {
            VirtualSocketType::Socketpair => "socketpair",
            VirtualSocketType::PipeRead | VirtualSocketType::PipeWrite => "pipe",
        }
    }

    fn metadata<'tcx>(
        &self,
    ) -> InterpResult<'tcx, Either<io::Result<std::fs::Metadata>, &'static str>> {
        let mode_name = match self.fd_type {
            VirtualSocketType::Socketpair => "S_IFSOCK",
            VirtualSocketType::PipeRead | VirtualSocketType::PipeWrite => "S_IFIFO",
        };
        interp_ok(Either::Right(mode_name))
    }

    fn destroy<'tcx>(
        self,
        _self_id: FdId,
        _communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        if let Some(peer_fd) = self.peer_fd().upgrade() {
            // If the current readbuf is non-empty when the file description is closed,
            // notify the peer that data lost has happened in current file description.
            if let Some(readbuf) = &self.readbuf {
                if !readbuf.borrow().buf.is_empty() {
                    peer_fd.peer_lost_data.set(true);
                }
            }
            // Notify peer fd that close has happened, since that can unblock reads and writes.
            ecx.update_fd_readiness(peer_fd, /* force_edge */ false)?;
        }
        interp_ok(Ok(()))
    }

    fn read<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        ecx.virtual_socket_read(self, ptr, len, /* is_non_block */ false, finish)
    }

    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        ecx.virtual_socket_write(self, ptr, len, /* is_non_block */ false, finish)
    }

    fn short_fd_operations(&self) -> bool {
        // Linux guarantees that when a read/write on a streaming socket comes back short,
        // the kernel buffer is empty/full:
        // See <https://man7.org/linux/man-pages/man7/epoll.7.html> in Q&A section.
        // So we can't do short reads/writes here.
        false
    }

    fn as_unix<'tcx>(
        self: FileDescriptionRef<Self>,
        _ecx: &MiriInterpCx<'tcx>,
    ) -> FileDescriptionRef<dyn UnixFileDescription> {
        self
    }

    fn get_flags<'tcx>(&self, ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, Scalar> {
        let mut flags = 0;

        // Get flag for file access mode.
        // The flag for both socketpair and pipe will remain the same even when the peer
        // fd is closed, so we need to look at the original type of this socket, not at whether
        // the peer socket still exists.
        match self.fd_type {
            VirtualSocketType::Socketpair => {
                flags |= ecx.eval_libc_i32("O_RDWR");
            }
            VirtualSocketType::PipeRead => {
                flags |= ecx.eval_libc_i32("O_RDONLY");
            }
            VirtualSocketType::PipeWrite => {
                flags |= ecx.eval_libc_i32("O_WRONLY");
            }
        }

        // Get flag for blocking status.
        if self.is_nonblock.get() {
            flags |= ecx.eval_libc_i32("O_NONBLOCK");
        }

        interp_ok(Scalar::from_i32(flags))
    }

    fn set_flags<'tcx>(
        &self,
        mut flag: i32,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let o_nonblock = ecx.eval_libc_i32("O_NONBLOCK");

        // O_NONBLOCK flag can be set / unset by user.
        if flag & o_nonblock == o_nonblock {
            self.is_nonblock.set(true);
            flag &= !o_nonblock;
        } else {
            self.is_nonblock.set(false);
        }

        // Throw error if there is any unsupported flag.
        if flag != 0 {
            throw_unsup_format!(
                "fcntl: only O_NONBLOCK is supported for F_SETFL on socketpairs and pipes"
            )
        }

        interp_ok(Scalar::from_i32(0))
    }

    fn readiness<'tcx>(&self) -> InterpResult<'tcx, Readiness> {
        // We only check the "readable", "writable", "read closed" and "write closed" readiness.
        // If other event flags need to be supported in the future, the check should be added here.

        let mut readiness = Readiness::EMPTY;

        // Check if it is readable.
        if let Some(readbuf) = &self.readbuf {
            if !readbuf.borrow().buf.is_empty() {
                readiness.readable = true;
            }
        } else {
            // Without a read buffer, reading never blocks, so we are always ready.
            readiness.readable = true;
        }

        // Check if is writable.
        if let Some(peer_fd) = self.peer_fd().upgrade() {
            if let Some(writebuf) = &peer_fd.readbuf {
                let data_size = writebuf.borrow().buf.len();
                let available_space = MAX_SOCKETPAIR_BUFFER_CAPACITY.strict_sub(data_size);
                if available_space != 0 {
                    readiness.writable = true;
                }
            } else {
                // Without a write buffer, writing never blocks.
                readiness.writable = true;
            }
        } else {
            // Peer FD has been closed. This always sets both the "read closed" and "write closed" flags
            // as we do not support `shutdown` that could be used to partially close the stream.
            readiness.read_closed = true;
            readiness.write_closed = true;
            // Since the peer is closed, even if no data is available reads will return EOF and
            // writes will return EPIPE. In other words, they won't block, so we mark this as ready
            // for read and write.
            readiness.readable = true;
            readiness.writable = true;
            // If there is data lost in peer_fd, set error readiness.
            if self.peer_lost_data.get() {
                readiness.error = true;
            }
        }
        interp_ok(readiness)
    }
}

impl UnixFileDescription for VirtualSocket {
    fn ioctl<'tcx>(
        &self,
        op: Scalar,
        arg: Option<&OpTy<'tcx>>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, i32> {
        match self.fd_type {
            VirtualSocketType::Socketpair => { /* fall-through to below */ }
            VirtualSocketType::PipeRead | VirtualSocketType::PipeWrite => {
                // The standard library only uses ioctl for changing the blocking mode
                // of Unix sockets. Thus, since using ioctl isn't the preferred way of
                // changing the blocking mode, we don't support it on pipes.
                throw_unsup_format!("cannot use ioctl on pipe");
            }
        }

        let fionbio = ecx.eval_libc("FIONBIO");

        if op == fionbio {
            // On these OSes, Rust uses the ioctl, so we trust that it is reasonable and controls
            // the same internal flag as fcntl.
            if !matches!(ecx.tcx.sess.target.os, Os::Linux | Os::Android | Os::MacOs | Os::FreeBsd)
            {
                // FIONBIO cannot be used to change the blocking mode of a socket on solarish targets:
                // <https://github.com/rust-lang/rust/commit/dda5c97675b4f5b1f6fdab64606c8a1f21021b0a>
                // Since there might be more targets which do weird things with this option, we use
                // an allowlist instead of just denying solarish targets.
                throw_unsup_format!(
                    "ioctl: setting FIONBIO on sockets is unsupported on target {}",
                    ecx.tcx.sess.target.os
                );
            }

            let Some(value_ptr) = arg else {
                throw_ub_format!("ioctl: setting FIONBIO on sockets requires a third argument");
            };
            let value = ecx.deref_pointer_as(value_ptr, ecx.machine.layouts.i32)?;
            let non_block = ecx.read_scalar(&value)?.to_i32()? != 0;
            self.is_nonblock.set(non_block);
            return interp_ok(0);
        }

        throw_unsup_format!("ioctl: unsupported operation {op:#x} on socket");
    }

    fn as_socket<'tcx>(
        self: FileDescriptionRef<Self>,
        _ecx: &MiriInterpCx<'tcx>,
    ) -> Option<FileDescriptionRef<dyn UnixSocketFileDescription>> {
        match self.fd_type {
            VirtualSocketType::Socketpair => Some(self),
            VirtualSocketType::PipeRead | VirtualSocketType::PipeWrite => None,
        }
    }
}

impl UnixSocketFileDescription for VirtualSocket {
    fn send<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        is_non_block: bool,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        ecx.virtual_socket_write(self, ptr, len, is_non_block, finish)
    }

    fn recv<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        is_peek: bool,
        is_non_block: bool,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        if is_peek {
            throw_unsup_format!("socketpair: virtual sockets don't support peeking")
        }

        ecx.virtual_socket_read(self, ptr, len, is_non_block, finish)
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Attempt two write `len` bytes from the buffer pointed to by `ptr` into the
    /// virtual socket `socket`.
    /// `is_non_block` specifies whether the operation should be performed as if the
    /// socket was non-blocking.
    /// After a successful write, `finish` is called with the amount of bytes written.
    fn virtual_socket_write(
        &mut self,
        socket: FileDescriptionRef<VirtualSocket>,
        ptr: Pointer,
        len: usize,
        is_non_block: bool,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Always succeed on write size 0.
        // ("If count is zero and fd refers to a file other than a regular file, the results are not specified.")
        if len == 0 {
            return finish.call(this, Ok(0));
        }

        // We are writing to our peer's readbuf.
        let Some(peer_fd) = socket.peer_fd().upgrade() else {
            // If the upgrade from Weak to Rc fails, it indicates that all read ends have been
            // closed. It is an error to write even if there would be space.
            return finish.call(this, Err(ErrorKind::BrokenPipe.into()));
        };

        let Some(writebuf) = &peer_fd.readbuf else {
            // Writing to the read end of a pipe.
            return finish.call(this, Err(IoError::LibcError("EBADF")));
        };

        // Let's see if we can write.
        let available_space =
            MAX_SOCKETPAIR_BUFFER_CAPACITY.strict_sub(writebuf.borrow().buf.len());
        if available_space == 0 {
            if socket.is_nonblock.get() || is_non_block {
                // Non-blocking socketpair with a full buffer.
                return finish.call(this, Err(ErrorKind::WouldBlock.into()));
            } else {
                socket.blocked_write_tid.borrow_mut().push(this.active_thread());
                this.block_thread(
                    BlockReason::VirtualSocket,
                    None,
                    callback!(
                        @capture<'tcx> {
                            socket: FileDescriptionRef<VirtualSocket>,
                            ptr: Pointer,
                            len: usize,
                            is_non_block: bool,
                            finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                        }
                        |this, unblock: UnblockKind| {
                            assert_eq!(unblock, UnblockKind::Ready);
                            this.virtual_socket_write(socket, ptr, len, is_non_block, finish)
                        }
                    ),
                );
            }
        } else {
            // There is space to write!
            let mut writebuf = writebuf.borrow_mut();
            // Remember this clock so `read` can synchronize with us.
            this.release_clock(|clock| {
                writebuf.clock.join(clock);
            })?;
            // Do full write / partial write based on the space available.
            let write_size = len.min(available_space);
            let actual_write_size =
                this.write_to_host(&mut writebuf.buf, write_size, ptr)?.unwrap();
            assert_eq!(actual_write_size, write_size);

            // Need to stop accessing peer_fd so that it can be notified.
            drop(writebuf);

            // Unblock all threads that are currently blocked on peer_fd's read.
            let waiting_threads = std::mem::take(&mut *peer_fd.blocked_read_tid.borrow_mut());
            // FIXME: We can randomize the order of unblocking.
            for thread_id in waiting_threads {
                this.unblock_thread(thread_id, BlockReason::VirtualSocket)?;
            }
            // Notify readiness watchers: we might be no longer writable, peer might now be readable.
            // The notification to the peer seems to be always sent on Linux, even if the
            // FD was readable before.
            this.update_fd_readiness(socket, /* force_edge */ false)?;
            this.update_fd_readiness(peer_fd, /* force_edge */ true)?;

            return finish.call(this, Ok(write_size));
        }
        interp_ok(())
    }

    /// Attempt to read `len` bytes from the virtual socket `socket` into the buffer
    /// pointed to by `ptr`.
    /// `is_non_block` specifies whether the operation should be performed as if the
    /// socket was non-blocking.
    /// After a successful read, `finish` is called with the amount of bytes read.
    fn virtual_socket_read(
        &mut self,
        socket: FileDescriptionRef<VirtualSocket>,
        ptr: Pointer,
        len: usize,
        is_non_block: bool,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Always succeed on read size 0.
        if len == 0 {
            return finish.call(this, Ok(0));
        }

        let Some(readbuf) = &socket.readbuf else {
            // FIXME: This should return EBADF, but there's no nice way to do that as there's no
            // corresponding ErrorKind variant.
            throw_unsup_format!("reading from the write end of a pipe")
        };

        if readbuf.borrow_mut().buf.is_empty() {
            if socket.peer_fd().upgrade().is_none() {
                // Socketpair with no peer and empty buffer.
                // 0 bytes successfully read indicates end-of-file.
                return finish.call(this, Ok(0));
            } else if socket.is_nonblock.get() || is_non_block {
                // Non-blocking socketpair with writer and empty buffer.
                // https://linux.die.net/man/2/read
                // EAGAIN or EWOULDBLOCK can be returned for socket,
                // POSIX.1-2001 allows either error to be returned for this case.
                // Since there is no ErrorKind for EAGAIN, WouldBlock is used.
                return finish.call(this, Err(ErrorKind::WouldBlock.into()));
            } else {
                socket.blocked_read_tid.borrow_mut().push(this.active_thread());
                this.block_thread(
                    BlockReason::VirtualSocket,
                    None,
                    callback!(
                        @capture<'tcx> {
                            socket: FileDescriptionRef<VirtualSocket>,
                            ptr: Pointer,
                            len: usize,
                            is_non_block: bool,
                            finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                        }
                        |this, unblock: UnblockKind| {
                            assert_eq!(unblock, UnblockKind::Ready);
                            this.virtual_socket_read(socket, ptr, len, is_non_block, finish)
                        }
                    ),
                );
            }
        } else {
            // There's data to be read!
            let mut readbuf = readbuf.borrow_mut();
            // Synchronize with all previous writes to this buffer.
            // FIXME: this over-synchronizes; a more precise approach would be to
            // only sync with the writes whose data we will read.
            this.acquire_clock(&readbuf.clock)?;

            // Do full read / partial read based on the space available.
            // Conveniently, `read` exists on `VecDeque` and has exactly the desired behavior.
            let read_size = this.read_from_host(|buf| readbuf.buf.read(buf), len, ptr)?.unwrap();
            let readbuf_now_empty = readbuf.buf.is_empty();

            // Need to drop before others can access the readbuf again.
            drop(readbuf);

            // A notification should be provided for the peer file description even when it can
            // only write 1 byte. This implementation is not compliant with the actual Linux kernel
            // implementation. For optimization reasons, the kernel will only mark the file description
            // as "writable" when it can write more than a certain number of bytes. Since we
            // don't know what that *certain number* is, we will provide a notification every time
            // a read is successful. This might result in our readiness emulation providing more
            // events than the real system.
            if let Some(peer_fd) = socket.peer_fd().upgrade() {
                // Unblock all threads that are currently blocked on peer_fd's write.
                let waiting_threads = std::mem::take(&mut *peer_fd.blocked_write_tid.borrow_mut());
                // FIXME: We can randomize the order of unblocking.
                for thread_id in waiting_threads {
                    this.unblock_thread(thread_id, BlockReason::VirtualSocket)?;
                }
                // Notify readiness watchers: peer is now writable.
                // Linux seems to always notify the peer if the read buffer is now empty.
                // (Linux also does that if this was a "big" read, but to avoid some arbitrary
                // threshold, we do not match that.)
                this.update_fd_readiness(peer_fd, /* force_edge */ readbuf_now_empty)?;
            };
            // Notify readiness watchers: we might be no longer readable.
            this.update_fd_readiness(socket, /* force_edge */ false)?;

            return finish.call(this, Ok(read_size));
        }
        interp_ok(())
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// For more information on the arguments see the socketpair manpage:
    /// <https://linux.die.net/man/2/socketpair>
    fn socketpair(
        &mut self,
        domain: &OpTy<'tcx>,
        type_: &OpTy<'tcx>,
        protocol: &OpTy<'tcx>,
        sv: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let domain = this.read_scalar(domain)?.to_i32()?;
        let mut flags = this.read_scalar(type_)?.to_i32()?;
        let protocol = this.read_scalar(protocol)?.to_i32()?;
        // This is really a pointer to `[i32; 2]` but we use a ptr-to-first-element representation.
        let sv = this.deref_pointer_as(sv, this.machine.layouts.i32)?;

        let mut is_sock_nonblock = false;

        // Interpret the flag. Every flag we recognize is "subtracted" from `flags`, so
        // if there is anything left at the end, that's an unsupported flag.
        if matches!(
            this.tcx.sess.target.os,
            Os::Linux | Os::Android | Os::FreeBsd | Os::Solaris | Os::Illumos
        ) {
            // SOCK_NONBLOCK and SOCK_CLOEXEC only exist on Linux, Android, FreeBSD,
            // Solaris, and Illumos targets.
            let sock_nonblock = this.eval_libc_i32("SOCK_NONBLOCK");
            let sock_cloexec = this.eval_libc_i32("SOCK_CLOEXEC");
            if flags & sock_nonblock == sock_nonblock {
                is_sock_nonblock = true;
                flags &= !sock_nonblock;
            }
            if flags & sock_cloexec == sock_cloexec {
                flags &= !sock_cloexec;
            }
        }

        // Fail on unsupported input.
        // AF_UNIX and AF_LOCAL are synonyms, so we accept both in case
        // their values differ.
        if domain != this.eval_libc_i32("AF_UNIX") && domain != this.eval_libc_i32("AF_LOCAL") {
            throw_unsup_format!(
                "socketpair: domain {:#x} is unsupported, only AF_UNIX \
                                 and AF_LOCAL are allowed",
                domain
            );
        } else if flags != this.eval_libc_i32("SOCK_STREAM") {
            throw_unsup_format!(
                "socketpair: type {:#x} is unsupported, only SOCK_STREAM, \
                                 SOCK_CLOEXEC and SOCK_NONBLOCK are allowed",
                flags
            );
        } else if protocol != 0 {
            throw_unsup_format!(
                "socketpair: socket protocol {protocol} is unsupported, \
                                 only 0 is allowed",
            );
        }

        // Generate file descriptions.
        let fds = &mut this.machine.fds;
        let fd0 = fds.new_ref(VirtualSocket {
            readbuf: Some(RefCell::new(Buffer::new())),
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
            is_nonblock: Cell::new(is_sock_nonblock),
            fd_type: VirtualSocketType::Socketpair,
        });
        let fd1 = fds.new_ref(VirtualSocket {
            readbuf: Some(RefCell::new(Buffer::new())),
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
            is_nonblock: Cell::new(is_sock_nonblock),
            fd_type: VirtualSocketType::Socketpair,
        });

        // Make the file descriptions point to each other.
        fd0.peer_fd.set(FileDescriptionRef::downgrade(&fd1)).unwrap();
        fd1.peer_fd.set(FileDescriptionRef::downgrade(&fd0)).unwrap();

        // Insert the file description to the fd table, generating the file descriptors.
        let sv0 = fds.insert(fd0);
        let sv1 = fds.insert(fd1);

        // Return socketpair file descriptors to the caller.
        let sv0 = Scalar::from_int(sv0, sv.layout.size);
        let sv1 = Scalar::from_int(sv1, sv.layout.size);
        this.write_scalar(sv0, &sv)?;
        this.write_scalar(sv1, &sv.offset(sv.layout.size, sv.layout, this)?)?;

        interp_ok(Scalar::from_i32(0))
    }

    fn pipe2(
        &mut self,
        pipefd: &OpTy<'tcx>,
        flags: Option<&OpTy<'tcx>>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let pipefd = this.deref_pointer_as(pipefd, this.machine.layouts.i32)?;
        let mut flags = match flags {
            Some(flags) => this.read_scalar(flags)?.to_i32()?,
            None => 0,
        };

        let cloexec = this.eval_libc_i32("O_CLOEXEC");
        let o_nonblock = this.eval_libc_i32("O_NONBLOCK");

        // Interpret the flag. Every flag we recognize is "subtracted" from `flags`, so
        // if there is anything left at the end, that's an unsupported flag.
        let mut is_nonblock = false;
        if flags & o_nonblock == o_nonblock {
            is_nonblock = true;
            flags &= !o_nonblock;
        }
        // As usual we ignore CLOEXEC.
        if flags & cloexec == cloexec {
            flags &= !cloexec;
        }
        if flags != 0 {
            throw_unsup_format!("unsupported flags in `pipe2`");
        }

        // Generate file descriptions.
        // pipefd[0] refers to the read end of the pipe.
        let fds = &mut this.machine.fds;
        let fd0 = fds.new_ref(VirtualSocket {
            readbuf: Some(RefCell::new(Buffer::new())),
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
            is_nonblock: Cell::new(is_nonblock),
            fd_type: VirtualSocketType::PipeRead,
        });
        let fd1 = fds.new_ref(VirtualSocket {
            readbuf: None,
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
            is_nonblock: Cell::new(is_nonblock),
            fd_type: VirtualSocketType::PipeWrite,
        });

        // Make the file descriptions point to each other.
        fd0.peer_fd.set(FileDescriptionRef::downgrade(&fd1)).unwrap();
        fd1.peer_fd.set(FileDescriptionRef::downgrade(&fd0)).unwrap();

        // Insert the file description to the fd table, generating the file descriptors.
        let pipefd0 = fds.insert(fd0);
        let pipefd1 = fds.insert(fd1);

        // Return file descriptors to the caller.
        let pipefd0 = Scalar::from_int(pipefd0, pipefd.layout.size);
        let pipefd1 = Scalar::from_int(pipefd1, pipefd.layout.size);
        this.write_scalar(pipefd0, &pipefd)?;
        this.write_scalar(pipefd1, &pipefd.offset(pipefd.layout.size, pipefd.layout, this)?)?;

        interp_ok(Scalar::from_i32(0))
    }
}
