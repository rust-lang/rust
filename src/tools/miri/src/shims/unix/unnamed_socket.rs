//! This implements "anonymous" sockets, that do not correspond to anything on the host system and
//! are entirely implemented inside Miri.
//! We also use the same infrastructure to implement unnamed pipes.

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::VecDeque;
use std::io;
use std::io::ErrorKind;

use crate::concurrency::VClock;
use crate::shims::files::{
    EvalContextExt as _, FileDescription, FileDescriptionRef, WeakFileDescriptionRef,
};
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::linux_like::epoll::{EpollReadyEvents, EvalContextExt as _};
use crate::*;

/// The maximum capacity of the socketpair buffer in bytes.
/// This number is arbitrary as the value can always
/// be configured in the real system.
const MAX_SOCKETPAIR_BUFFER_CAPACITY: usize = 212992;

#[derive(Debug, PartialEq)]
enum AnonSocketType {
    // Either end of the socketpair fd.
    Socketpair,
    // Read end of the pipe.
    PipeRead,
    // Write end of the pipe.
    PipeWrite,
}

/// One end of a pair of connected unnamed sockets.
#[derive(Debug)]
struct AnonSocket {
    /// The buffer we are reading from, or `None` if this is the writing end of a pipe.
    /// (In that case, the peer FD will be the reading end of that pipe.)
    readbuf: Option<RefCell<Buffer>>,
    /// The `AnonSocket` file descriptor that is our "peer", and that holds the buffer we are
    /// writing to. This is a weak reference because the other side may be closed before us; all
    /// future writes will then trigger EPIPE.
    peer_fd: OnceCell<WeakFileDescriptionRef<AnonSocket>>,
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
    // Differentiate between different AnonSocket fd types.
    fd_type: AnonSocketType,
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

impl AnonSocket {
    fn peer_fd(&self) -> &WeakFileDescriptionRef<AnonSocket> {
        self.peer_fd.get().unwrap()
    }
}

impl FileDescription for AnonSocket {
    fn name(&self) -> &'static str {
        match self.fd_type {
            AnonSocketType::Socketpair => "socketpair",
            AnonSocketType::PipeRead | AnonSocketType::PipeWrite => "pipe",
        }
    }

    fn close<'tcx>(
        self,
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
            ecx.check_and_update_readiness(peer_fd)?;
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
        anonsocket_read(self, ptr, len, ecx, finish)
    }

    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        anonsocket_write(self, ptr, len, ecx, finish)
    }

    fn as_unix<'tcx>(&self, _ecx: &MiriInterpCx<'tcx>) -> &dyn UnixFileDescription {
        self
    }

    fn get_flags<'tcx>(&self, ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, Scalar> {
        let mut flags = 0;

        // Get flag for file access mode.
        // The flag for both socketpair and pipe will remain the same even when the peer
        // fd is closed, so we need to look at the original type of this socket, not at whether
        // the peer socket still exists.
        match self.fd_type {
            AnonSocketType::Socketpair => {
                flags |= ecx.eval_libc_i32("O_RDWR");
            }
            AnonSocketType::PipeRead => {
                flags |= ecx.eval_libc_i32("O_RDONLY");
            }
            AnonSocketType::PipeWrite => {
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
        // FIXME: File creation flags should be ignored.

        let o_nonblock = ecx.eval_libc_i32("O_NONBLOCK");
        let o_rdonly = ecx.eval_libc_i32("O_RDONLY");
        let o_wronly = ecx.eval_libc_i32("O_WRONLY");
        let o_rdwr = ecx.eval_libc_i32("O_RDWR");

        // O_NONBLOCK flag can be set / unset by user.
        if flag & o_nonblock == o_nonblock {
            self.is_nonblock.set(true);
            flag &= !o_nonblock;
        } else {
            self.is_nonblock.set(false);
        }

        // Ignore all file access mode flags.
        flag &= !(o_rdonly | o_wronly | o_rdwr);

        // Throw error if there is any unsupported flag.
        if flag != 0 {
            throw_unsup_format!(
                "fcntl: only O_NONBLOCK is supported for F_SETFL on socketpairs and pipes"
            )
        }

        interp_ok(Scalar::from_i32(0))
    }
}

/// Write to AnonSocket based on the space available and return the written byte size.
fn anonsocket_write<'tcx>(
    self_ref: FileDescriptionRef<AnonSocket>,
    ptr: Pointer,
    len: usize,
    ecx: &mut MiriInterpCx<'tcx>,
    finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
) -> InterpResult<'tcx> {
    // Always succeed on write size 0.
    // ("If count is zero and fd refers to a file other than a regular file, the results are not specified.")
    if len == 0 {
        return finish.call(ecx, Ok(0));
    }

    // We are writing to our peer's readbuf.
    let Some(peer_fd) = self_ref.peer_fd().upgrade() else {
        // If the upgrade from Weak to Rc fails, it indicates that all read ends have been
        // closed. It is an error to write even if there would be space.
        return finish.call(ecx, Err(ErrorKind::BrokenPipe.into()));
    };

    let Some(writebuf) = &peer_fd.readbuf else {
        // Writing to the read end of a pipe.
        return finish.call(ecx, Err(IoError::LibcError("EBADF")));
    };

    // Let's see if we can write.
    let available_space = MAX_SOCKETPAIR_BUFFER_CAPACITY.strict_sub(writebuf.borrow().buf.len());
    if available_space == 0 {
        if self_ref.is_nonblock.get() {
            // Non-blocking socketpair with a full buffer.
            return finish.call(ecx, Err(ErrorKind::WouldBlock.into()));
        } else {
            self_ref.blocked_write_tid.borrow_mut().push(ecx.active_thread());
            // Blocking socketpair with a full buffer.
            // Block the current thread; only keep a weak ref for this.
            let weak_self_ref = FileDescriptionRef::downgrade(&self_ref);
            ecx.block_thread(
                BlockReason::UnnamedSocket,
                None,
                callback!(
                    @capture<'tcx> {
                        weak_self_ref: WeakFileDescriptionRef<AnonSocket>,
                        ptr: Pointer,
                        len: usize,
                        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                    }
                    |this, unblock: UnblockKind| {
                        assert_eq!(unblock, UnblockKind::Ready);
                        // If we got unblocked, then our peer successfully upgraded its weak
                        // ref to us. That means we can also upgrade our weak ref.
                        let self_ref = weak_self_ref.upgrade().unwrap();
                        anonsocket_write(self_ref, ptr, len, this, finish)
                    }
                ),
            );
        }
    } else {
        // There is space to write!
        let mut writebuf = writebuf.borrow_mut();
        // Remember this clock so `read` can synchronize with us.
        ecx.release_clock(|clock| {
            writebuf.clock.join(clock);
        });
        // Do full write / partial write based on the space available.
        let write_size = len.min(available_space);
        let actual_write_size = ecx.write_to_host(&mut writebuf.buf, write_size, ptr)?.unwrap();
        assert_eq!(actual_write_size, write_size);

        // Need to stop accessing peer_fd so that it can be notified.
        drop(writebuf);

        // Unblock all threads that are currently blocked on peer_fd's read.
        let waiting_threads = std::mem::take(&mut *peer_fd.blocked_read_tid.borrow_mut());
        // FIXME: We can randomize the order of unblocking.
        for thread_id in waiting_threads {
            ecx.unblock_thread(thread_id, BlockReason::UnnamedSocket)?;
        }
        // Notification should be provided for peer fd as it became readable.
        // The kernel does this even if the fd was already readable before, so we follow suit.
        ecx.check_and_update_readiness(peer_fd)?;

        return finish.call(ecx, Ok(write_size));
    }
    interp_ok(())
}

/// Read from AnonSocket and return the number of bytes read.
fn anonsocket_read<'tcx>(
    self_ref: FileDescriptionRef<AnonSocket>,
    ptr: Pointer,
    len: usize,
    ecx: &mut MiriInterpCx<'tcx>,
    finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
) -> InterpResult<'tcx> {
    // Always succeed on read size 0.
    if len == 0 {
        return finish.call(ecx, Ok(0));
    }

    let Some(readbuf) = &self_ref.readbuf else {
        // FIXME: This should return EBADF, but there's no nice way to do that as there's no
        // corresponding ErrorKind variant.
        throw_unsup_format!("reading from the write end of a pipe")
    };

    if readbuf.borrow_mut().buf.is_empty() {
        if self_ref.peer_fd().upgrade().is_none() {
            // Socketpair with no peer and empty buffer.
            // 0 bytes successfully read indicates end-of-file.
            return finish.call(ecx, Ok(0));
        } else if self_ref.is_nonblock.get() {
            // Non-blocking socketpair with writer and empty buffer.
            // https://linux.die.net/man/2/read
            // EAGAIN or EWOULDBLOCK can be returned for socket,
            // POSIX.1-2001 allows either error to be returned for this case.
            // Since there is no ErrorKind for EAGAIN, WouldBlock is used.
            return finish.call(ecx, Err(ErrorKind::WouldBlock.into()));
        } else {
            self_ref.blocked_read_tid.borrow_mut().push(ecx.active_thread());
            // Blocking socketpair with writer and empty buffer.
            // Block the current thread; only keep a weak ref for this.
            let weak_self_ref = FileDescriptionRef::downgrade(&self_ref);
            ecx.block_thread(
                BlockReason::UnnamedSocket,
                None,
                callback!(
                    @capture<'tcx> {
                        weak_self_ref: WeakFileDescriptionRef<AnonSocket>,
                        ptr: Pointer,
                        len: usize,
                        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                    }
                    |this, unblock: UnblockKind| {
                        assert_eq!(unblock, UnblockKind::Ready);
                        // If we got unblocked, then our peer successfully upgraded its weak
                        // ref to us. That means we can also upgrade our weak ref.
                        let self_ref = weak_self_ref.upgrade().unwrap();
                        anonsocket_read(self_ref, ptr, len, this, finish)
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
        ecx.acquire_clock(&readbuf.clock);

        // Do full read / partial read based on the space available.
        // Conveniently, `read` exists on `VecDeque` and has exactly the desired behavior.
        let read_size = ecx.read_from_host(&mut readbuf.buf, len, ptr)?.unwrap();

        // Need to drop before others can access the readbuf again.
        drop(readbuf);

        // A notification should be provided for the peer file description even when it can
        // only write 1 byte. This implementation is not compliant with the actual Linux kernel
        // implementation. For optimization reasons, the kernel will only mark the file description
        // as "writable" when it can write more than a certain number of bytes. Since we
        // don't know what that *certain number* is, we will provide a notification every time
        // a read is successful. This might result in our epoll emulation providing more
        // notifications than the real system.
        if let Some(peer_fd) = self_ref.peer_fd().upgrade() {
            // Unblock all threads that are currently blocked on peer_fd's write.
            let waiting_threads = std::mem::take(&mut *peer_fd.blocked_write_tid.borrow_mut());
            // FIXME: We can randomize the order of unblocking.
            for thread_id in waiting_threads {
                ecx.unblock_thread(thread_id, BlockReason::UnnamedSocket)?;
            }
            // Notify epoll waiters.
            ecx.check_and_update_readiness(peer_fd)?;
        };

        return finish.call(ecx, Ok(read_size));
    }
    interp_ok(())
}

impl UnixFileDescription for AnonSocket {
    fn get_epoll_ready_events<'tcx>(&self) -> InterpResult<'tcx, EpollReadyEvents> {
        // We only check the status of EPOLLIN, EPOLLOUT, EPOLLHUP and EPOLLRDHUP flags.
        // If other event flags need to be supported in the future, the check should be added here.

        let mut epoll_ready_events = EpollReadyEvents::new();

        // Check if it is readable.
        if let Some(readbuf) = &self.readbuf {
            if !readbuf.borrow().buf.is_empty() {
                epoll_ready_events.epollin = true;
            }
        } else {
            // Without a read buffer, reading never blocks, so we are always ready.
            epoll_ready_events.epollin = true;
        }

        // Check if is writable.
        if let Some(peer_fd) = self.peer_fd().upgrade() {
            if let Some(writebuf) = &peer_fd.readbuf {
                let data_size = writebuf.borrow().buf.len();
                let available_space = MAX_SOCKETPAIR_BUFFER_CAPACITY.strict_sub(data_size);
                if available_space != 0 {
                    epoll_ready_events.epollout = true;
                }
            } else {
                // Without a write buffer, writing never blocks.
                epoll_ready_events.epollout = true;
            }
        } else {
            // Peer FD has been closed. This always sets both the RDHUP and HUP flags
            // as we do not support `shutdown` that could be used to partially close the stream.
            epoll_ready_events.epollrdhup = true;
            epoll_ready_events.epollhup = true;
            // Since the peer is closed, even if no data is available reads will return EOF and
            // writes will return EPIPE. In other words, they won't block, so we mark this as ready
            // for read and write.
            epoll_ready_events.epollin = true;
            epoll_ready_events.epollout = true;
            // If there is data lost in peer_fd, set EPOLLERR.
            if self.peer_lost_data.get() {
                epoll_ready_events.epollerr = true;
            }
        }
        interp_ok(epoll_ready_events)
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
        if this.tcx.sess.target.os == "linux" {
            // SOCK_NONBLOCK only exists on Linux.
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
        let fd0 = fds.new_ref(AnonSocket {
            readbuf: Some(RefCell::new(Buffer::new())),
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
            is_nonblock: Cell::new(is_sock_nonblock),
            fd_type: AnonSocketType::Socketpair,
        });
        let fd1 = fds.new_ref(AnonSocket {
            readbuf: Some(RefCell::new(Buffer::new())),
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
            is_nonblock: Cell::new(is_sock_nonblock),
            fd_type: AnonSocketType::Socketpair,
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
        let fd0 = fds.new_ref(AnonSocket {
            readbuf: Some(RefCell::new(Buffer::new())),
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
            is_nonblock: Cell::new(is_nonblock),
            fd_type: AnonSocketType::PipeRead,
        });
        let fd1 = fds.new_ref(AnonSocket {
            readbuf: None,
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
            is_nonblock: Cell::new(is_nonblock),
            fd_type: AnonSocketType::PipeWrite,
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
