//! This implements "anonymous" sockets, that do not correspond to anything on the host system and
//! are entirely implemented inside Miri.
//! We also use the same infrastructure to implement unnamed pipes.

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::VecDeque;
use std::io;
use std::io::{ErrorKind, Read};

use rustc_target::abi::Size;

use crate::concurrency::VClock;
use crate::shims::unix::fd::{FileDescriptionRef, WeakFileDescriptionRef};
use crate::shims::unix::linux::epoll::{EpollReadyEvents, EvalContextExt as _};
use crate::shims::unix::*;
use crate::*;

/// The maximum capacity of the socketpair buffer in bytes.
/// This number is arbitrary as the value can always
/// be configured in the real system.
const MAX_SOCKETPAIR_BUFFER_CAPACITY: usize = 212992;

/// One end of a pair of connected unnamed sockets.
#[derive(Debug)]
struct AnonSocket {
    /// The buffer we are reading from, or `None` if this is the writing end of a pipe.
    /// (In that case, the peer FD will be the reading end of that pipe.)
    readbuf: Option<RefCell<Buffer>>,
    /// The `AnonSocket` file descriptor that is our "peer", and that holds the buffer we are
    /// writing to. This is a weak reference because the other side may be closed before us; all
    /// future writes will then trigger EPIPE.
    peer_fd: OnceCell<WeakFileDescriptionRef>,
    /// Indicates whether the peer has lost data when the file description is closed.
    /// This flag is set to `true` if the peer's `readbuf` is non-empty at the time
    /// of closure.
    peer_lost_data: Cell<bool>,
    is_nonblock: bool,
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
    fn peer_fd(&self) -> &WeakFileDescriptionRef {
        self.peer_fd.get().unwrap()
    }
}

impl FileDescription for AnonSocket {
    fn name(&self) -> &'static str {
        "socketpair"
    }

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
            if let Some(writebuf) = &peer_fd.downcast::<AnonSocket>().unwrap().readbuf {
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

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        if let Some(peer_fd) = self.peer_fd().upgrade() {
            // If the current readbuf is non-empty when the file description is closed,
            // notify the peer that data lost has happened in current file description.
            if let Some(readbuf) = &self.readbuf {
                if !readbuf.borrow().buf.is_empty() {
                    peer_fd.downcast::<AnonSocket>().unwrap().peer_lost_data.set(true);
                }
            }
            // Notify peer fd that close has happened, since that can unblock reads and writes.
            ecx.check_and_update_readiness(&peer_fd)?;
        }
        interp_ok(Ok(()))
    }

    fn read<'tcx>(
        &self,
        _self_ref: &FileDescriptionRef,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        let mut bytes = vec![0; len];

        // Always succeed on read size 0.
        if len == 0 {
            return ecx.return_read_success(ptr, &bytes, 0, dest);
        }

        let Some(readbuf) = &self.readbuf else {
            // FIXME: This should return EBADF, but there's no nice way to do that as there's no
            // corresponding ErrorKind variant.
            throw_unsup_format!("reading from the write end of a pipe");
        };
        let mut readbuf = readbuf.borrow_mut();
        if readbuf.buf.is_empty() {
            if self.peer_fd().upgrade().is_none() {
                // Socketpair with no peer and empty buffer.
                // 0 bytes successfully read indicates end-of-file.
                return ecx.return_read_success(ptr, &bytes, 0, dest);
            } else {
                if self.is_nonblock {
                    // Non-blocking socketpair with writer and empty buffer.
                    // https://linux.die.net/man/2/read
                    // EAGAIN or EWOULDBLOCK can be returned for socket,
                    // POSIX.1-2001 allows either error to be returned for this case.
                    // Since there is no ErrorKind for EAGAIN, WouldBlock is used.
                    return ecx.set_last_error_and_return(ErrorKind::WouldBlock, dest);
                } else {
                    // Blocking socketpair with writer and empty buffer.
                    // FIXME: blocking is currently not supported
                    throw_unsup_format!("socketpair read: blocking isn't supported yet");
                }
            }
        }

        // Synchronize with all previous writes to this buffer.
        // FIXME: this over-synchronizes; a more precise approach would be to
        // only sync with the writes whose data we will read.
        ecx.acquire_clock(&readbuf.clock);

        // Do full read / partial read based on the space available.
        // Conveniently, `read` exists on `VecDeque` and has exactly the desired behavior.
        let actual_read_size = readbuf.buf.read(&mut bytes).unwrap();

        // Need to drop before others can access the readbuf again.
        drop(readbuf);

        // A notification should be provided for the peer file description even when it can
        // only write 1 byte. This implementation is not compliant with the actual Linux kernel
        // implementation. For optimization reasons, the kernel will only mark the file description
        // as "writable" when it can write more than a certain number of bytes. Since we
        // don't know what that *certain number* is, we will provide a notification every time
        // a read is successful. This might result in our epoll emulation providing more
        // notifications than the real system.
        if let Some(peer_fd) = self.peer_fd().upgrade() {
            ecx.check_and_update_readiness(&peer_fd)?;
        }

        ecx.return_read_success(ptr, &bytes, actual_read_size, dest)
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
        // Always succeed on write size 0.
        // ("If count is zero and fd refers to a file other than a regular file, the results are not specified.")
        if len == 0 {
            return ecx.return_write_success(0, dest);
        }

        // We are writing to our peer's readbuf.
        let Some(peer_fd) = self.peer_fd().upgrade() else {
            // If the upgrade from Weak to Rc fails, it indicates that all read ends have been
            // closed.
            return ecx.set_last_error_and_return(ErrorKind::BrokenPipe, dest);
        };

        let Some(writebuf) = &peer_fd.downcast::<AnonSocket>().unwrap().readbuf else {
            // FIXME: This should return EBADF, but there's no nice way to do that as there's no
            // corresponding ErrorKind variant.
            throw_unsup_format!("writing to the reading end of a pipe");
        };
        let mut writebuf = writebuf.borrow_mut();
        let data_size = writebuf.buf.len();
        let available_space = MAX_SOCKETPAIR_BUFFER_CAPACITY.strict_sub(data_size);
        if available_space == 0 {
            if self.is_nonblock {
                // Non-blocking socketpair with a full buffer.
                return ecx.set_last_error_and_return(ErrorKind::WouldBlock, dest);
            } else {
                // Blocking socketpair with a full buffer.
                throw_unsup_format!("socketpair write: blocking isn't supported yet");
            }
        }
        // Remember this clock so `read` can synchronize with us.
        if let Some(clock) = &ecx.release_clock() {
            writebuf.clock.join(clock);
        }
        // Do full write / partial write based on the space available.
        let actual_write_size = len.min(available_space);
        let bytes = ecx.read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(len))?;
        writebuf.buf.extend(&bytes[..actual_write_size]);

        // Need to stop accessing peer_fd so that it can be notified.
        drop(writebuf);

        // Notification should be provided for peer fd as it became readable.
        // The kernel does this even if the fd was already readable before, so we follow suit.
        ecx.check_and_update_readiness(&peer_fd)?;

        ecx.return_write_success(actual_write_size, dest)
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
        let mut type_ = this.read_scalar(type_)?.to_i32()?;
        let protocol = this.read_scalar(protocol)?.to_i32()?;
        let sv = this.deref_pointer(sv)?;

        let mut is_sock_nonblock = false;

        // Parse and remove the type flags that we support.
        // SOCK_NONBLOCK only exists on Linux.
        if this.tcx.sess.target.os == "linux" {
            if type_ & this.eval_libc_i32("SOCK_NONBLOCK") == this.eval_libc_i32("SOCK_NONBLOCK") {
                is_sock_nonblock = true;
                type_ &= !(this.eval_libc_i32("SOCK_NONBLOCK"));
            }
            if type_ & this.eval_libc_i32("SOCK_CLOEXEC") == this.eval_libc_i32("SOCK_CLOEXEC") {
                type_ &= !(this.eval_libc_i32("SOCK_CLOEXEC"));
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
        } else if type_ != this.eval_libc_i32("SOCK_STREAM") {
            throw_unsup_format!(
                "socketpair: type {:#x} is unsupported, only SOCK_STREAM, \
                                 SOCK_CLOEXEC and SOCK_NONBLOCK are allowed",
                type_
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
            is_nonblock: is_sock_nonblock,
        });
        let fd1 = fds.new_ref(AnonSocket {
            readbuf: Some(RefCell::new(Buffer::new())),
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            is_nonblock: is_sock_nonblock,
        });

        // Make the file descriptions point to each other.
        fd0.downcast::<AnonSocket>().unwrap().peer_fd.set(fd1.downgrade()).unwrap();
        fd1.downcast::<AnonSocket>().unwrap().peer_fd.set(fd0.downgrade()).unwrap();

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
        let flags = match flags {
            Some(flags) => this.read_scalar(flags)?.to_i32()?,
            None => 0,
        };

        // As usual we ignore CLOEXEC.
        let cloexec = this.eval_libc_i32("O_CLOEXEC");
        if flags != 0 && flags != cloexec {
            throw_unsup_format!("unsupported flags in `pipe2`");
        }

        // Generate file descriptions.
        // pipefd[0] refers to the read end of the pipe.
        let fds = &mut this.machine.fds;
        let fd0 = fds.new_ref(AnonSocket {
            readbuf: Some(RefCell::new(Buffer::new())),
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            is_nonblock: false,
        });
        let fd1 = fds.new_ref(AnonSocket {
            readbuf: None,
            peer_fd: OnceCell::new(),
            peer_lost_data: Cell::new(false),
            is_nonblock: false,
        });

        // Make the file descriptions point to each other.
        fd0.downcast::<AnonSocket>().unwrap().peer_fd.set(fd1.downgrade()).unwrap();
        fd1.downcast::<AnonSocket>().unwrap().peer_fd.set(fd0.downgrade()).unwrap();

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
