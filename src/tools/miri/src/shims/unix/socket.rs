use std::cell::{Cell, RefCell, RefMut};
use std::io;
use std::io::Read;
use std::net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use mio::event::Source;
use mio::net::{TcpListener, TcpStream};
use rustc_abi::Size;
use rustc_const_eval::interpret::{InterpResult, interp_ok};
use rustc_middle::throw_unsup_format;
use rustc_target::spec::Os;

use crate::shims::files::{EvalContextExt as _, FdId, FileDescription, FileDescriptionRef};
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::linux_like::epoll::{EpollReadiness, EvalContextExt as _};
use crate::shims::unix::socket_address::EvalContextExt as _;
use crate::*;

#[derive(Debug, PartialEq)]
enum SocketFamily {
    // IPv4 internet protocols
    IPv4,
    // IPv6 internet protocols
    IPv6,
}

#[derive(Debug)]
enum SocketState {
    /// No syscall after `socket` has been made.
    Initial,
    /// The `bind` syscall has been called on the socket.
    /// This is only reachable from the [`SocketState::Initial`] state.
    Bound(SocketAddr),
    /// The `listen` syscall has been called on the socket.
    /// This is only reachable from the [`SocketState::Bound`] state.
    Listening(TcpListener),
    /// The `connect` syscall has been called and we weren't yet able
    /// to ensure the connection is established. This is only reachable
    /// from the [`SocketState::Initial`] state.
    Connecting(TcpStream),
    /// The `connect` syscall has been called on the socket and
    /// we ensured that the connection is established, or
    /// the socket was created by the `accept` syscall.
    /// For a socket created using the `connect` syscall, this is
    /// only reachable from the [`SocketState::Connecting`] state.
    Connected(TcpStream),
}

#[derive(Debug)]
struct Socket {
    /// Family of the socket, used to ensure socket only binds/connects to address of
    /// same family.
    family: SocketFamily,
    /// Current state of the inner socket.
    state: RefCell<SocketState>,
    /// Whether this fd is non-blocking or not.
    is_non_block: Cell<bool>,
    /// The current blocking I/O readiness of the file description.
    io_readiness: RefCell<BlockingIoSourceReadiness>,
    /// [`Some`] when the socket had an async error which has not yet been fetched via `SO_ERROR`.
    error: RefCell<Option<io::Error>>,
}

impl FileDescription for Socket {
    fn name(&self) -> &'static str {
        "socket"
    }

    fn destroy<'tcx>(
        self,
        self_id: FdId,
        communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        assert!(communicate_allowed, "cannot have `Socket` with isolation enabled!");

        if matches!(
            &*self.state.borrow(),
            SocketState::Listening(_) | SocketState::Connecting(_) | SocketState::Connected(_)
        ) {
            // There exists an associated host socket so we need to deregister it
            // from the blocking I/O manager.
            ecx.machine.blocking_io.deregister(self_id, self)
        };

        interp_ok(Ok(()))
    }

    fn read<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `Socket` with isolation enabled!");

        let socket = self;

        ecx.ensure_connected(
            socket.clone(),
            !socket.is_non_block.get(),
            "read",
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<Socket>,
                    ptr: Pointer,
                    len: usize,
                    finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                } |this, result: Result<(), ()>| {
                    if result.is_err() {
                        return finish.call(this, Err(LibcError("ENOTCONN")))
                    }

                    // Since `read` is the same as `recv` with no flags, we just treat
                    // the `read` as a `recv` here.

                    if socket.is_non_block.get() {
                        // We have a non-blocking socket and thus don't want to block until
                        // we can read.
                        let result = this.try_non_block_recv(&socket, ptr, len, /* should_peek */ false)?;
                        finish.call(this, result)
                    } else {
                        // The socket is in blocking mode and thus the read call should block
                        // until we can read some bytes from the socket.
                        this.block_for_recv(socket, ptr, len, /* should_peek */ false, finish)
                    }
                }
            ),
        )
    }

    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `Socket` with isolation enabled!");

        let socket = self;

        ecx.ensure_connected(
            socket.clone(),
            !socket.is_non_block.get(),
            "write",
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<Socket>,
                    ptr: Pointer,
                    len: usize,
                    finish: DynMachineCallback<'tcx, Result<usize, IoError>>
                } |this, result: Result<(), ()>| {
                    if result.is_err() {
                        return finish.call(this, Err(LibcError("ENOTCONN")))
                    }

                    // Since `write` is the same as `send` with no flags, we just treat
                    // the `write` as a `send` here.

                    if socket.is_non_block.get() {
                        // We have a non-blocking socket and thus don't want to block until
                        // we can write.
                        let result = this.try_non_block_send(&socket, ptr, len)?;
                        return finish.call(this, result)
                    } else {
                        // The socket is in blocking mode and thus the write call should block
                        // until we can write some bytes into the socket.
                        this.block_for_send(socket, ptr, len, finish)
                    }
                }
            ),
        )
    }

    fn short_fd_operations(&self) -> bool {
        // Linux guarantees that when a read/write on a streaming socket comes back short,
        // the kernel buffer is empty/full:
        // See <https://man7.org/linux/man-pages/man7/epoll.7.html> in Q&A section.
        // So we can't do short reads/writes here.
        false
    }

    fn as_unix<'tcx>(&self, _ecx: &MiriInterpCx<'tcx>) -> &dyn UnixFileDescription {
        self
    }

    fn get_flags<'tcx>(&self, ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, Scalar> {
        let mut flags = ecx.eval_libc_i32("O_RDWR");

        if self.is_non_block.get() {
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
            self.is_non_block.set(true);
            flag &= !o_nonblock;
        } else {
            self.is_non_block.set(false);
        }

        // Throw error if there is any unsupported flag.
        if flag != 0 {
            throw_unsup_format!("fcntl: only O_NONBLOCK is supported for sockets")
        }

        interp_ok(Scalar::from_i32(0))
    }
}

impl UnixFileDescription for Socket {
    fn ioctl<'tcx>(
        &self,
        op: Scalar,
        arg: Option<&OpTy<'tcx>>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, i32> {
        assert!(ecx.machine.communicate(), "cannot have `Socket` with isolation enabled!");

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
            self.is_non_block.set(non_block);
            return interp_ok(0);
        }

        throw_unsup_format!("ioctl: unsupported operation {op:#x} on socket");
    }

    fn epoll_active_events<'tcx>(&self) -> InterpResult<'tcx, EpollReadiness> {
        interp_ok(EpollReadiness::from(&*self.io_readiness.borrow()))
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// For more information on the arguments see the socket manpage:
    /// <https://linux.die.net/man/2/socket>
    fn socket(
        &mut self,
        domain: &OpTy<'tcx>,
        type_: &OpTy<'tcx>,
        protocol: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let domain = this.read_scalar(domain)?.to_i32()?;
        let mut flags = this.read_scalar(type_)?.to_i32()?;
        let protocol = this.read_scalar(protocol)?.to_i32()?;

        // Reject if isolation is enabled
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`socket`", reject_with)?;
            return this.set_last_error_and_return_i32(LibcError("EACCES"));
        }

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
                // We don't support `exec` so we can ignore this.
                flags &= !sock_cloexec;
            }
        }

        let family = if domain == this.eval_libc_i32("AF_INET") {
            SocketFamily::IPv4
        } else if domain == this.eval_libc_i32("AF_INET6") {
            SocketFamily::IPv6
        } else {
            throw_unsup_format!(
                "socket: domain {:#x} is unsupported, only AF_INET and \
                AF_INET6 are allowed.",
                domain
            );
        };

        if flags != this.eval_libc_i32("SOCK_STREAM") {
            throw_unsup_format!(
                "socket: type {:#x} is unsupported, only SOCK_STREAM, \
                SOCK_CLOEXEC and SOCK_NONBLOCK are allowed",
                flags
            );
        }
        if protocol != 0 && protocol != this.eval_libc_i32("IPPROTO_TCP") {
            throw_unsup_format!(
                "socket: socket protocol {protocol} is unsupported, \
                only IPPROTO_TCP and 0 are allowed"
            );
        }

        let fds = &mut this.machine.fds;
        let fd = fds.new_ref(Socket {
            family,
            state: RefCell::new(SocketState::Initial),
            is_non_block: Cell::new(is_sock_nonblock),
            io_readiness: RefCell::new(BlockingIoSourceReadiness::empty()),
            error: RefCell::new(None),
        });

        interp_ok(Scalar::from_i32(fds.insert(fd)))
    }

    fn bind(
        &mut self,
        socket: &OpTy<'tcx>,
        address: &OpTy<'tcx>,
        address_len: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let address = match this.read_socket_address(address, address_len, "bind")? {
            Ok(addr) => addr,
            Err(e) => return this.set_last_error_and_return_i32(e),
        };

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return this.set_last_error_and_return_i32(LibcError("ENOTSOCK"));
        };

        assert!(this.machine.communicate(), "cannot have `Socket` with isolation enabled!");

        let mut state = socket.state.borrow_mut();

        match *state {
            SocketState::Initial => {
                let address_family = match &address {
                    SocketAddr::V4(_) => SocketFamily::IPv4,
                    SocketAddr::V6(_) => SocketFamily::IPv6,
                };

                if socket.family != address_family {
                    // Attempted to bind an address from a family that doesn't match
                    // the family of the socket.
                    let err = if matches!(this.tcx.sess.target.os, Os::Linux | Os::Android) {
                        // Linux man page states that `EINVAL` is used when there is an address family mismatch.
                        // See <https://man7.org/linux/man-pages/man2/bind.2.html>
                        LibcError("EINVAL")
                    } else {
                        // POSIX man page states that `EAFNOSUPPORT` should be used when there is an address
                        // family mismatch.
                        // See <https://man7.org/linux/man-pages/man3/bind.3p.html>
                        LibcError("EAFNOSUPPORT")
                    };
                    return this.set_last_error_and_return_i32(err);
                }

                *state = SocketState::Bound(address);
            }
            SocketState::Connecting(_) | SocketState::Connected(_) =>
                throw_unsup_format!(
                    "bind: socket is already connected and binding a
                    connected socket is unsupported"
                ),
            SocketState::Bound(_) | SocketState::Listening(_) =>
                throw_unsup_format!(
                    "bind: socket is already bound and binding a socket \
                    multiple times is unsupported"
                ),
        }

        interp_ok(Scalar::from_i32(0))
    }

    fn listen(&mut self, socket: &OpTy<'tcx>, backlog: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        // Since the backlog value is just a performance hint we can ignore it.
        let _backlog = this.read_scalar(backlog)?.to_i32()?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return this.set_last_error_and_return_i32(LibcError("ENOTSOCK"));
        };

        assert!(this.machine.communicate(), "cannot have `Socket` with isolation enabled!");

        let mut state = socket.state.borrow_mut();

        match *state {
            SocketState::Bound(socket_addr) =>
                match TcpListener::bind(socket_addr) {
                    Ok(listener) => {
                        *state = SocketState::Listening(listener);
                        drop(state);
                        // Register the socket to the blocking I/O manager because
                        // we now have an associated host socket.
                        this.machine.blocking_io.register(socket);
                    }
                    Err(e) => return this.set_last_error_and_return_i32(e),
                },
            SocketState::Initial => {
                throw_unsup_format!(
                    "listen: listening on a socket which isn't bound is unsupported"
                )
            }
            SocketState::Listening(_) => {
                throw_unsup_format!("listen: listening on a socket multiple times is unsupported")
            }
            SocketState::Connecting(_) | SocketState::Connected(_) => {
                throw_unsup_format!("listen: listening on a connected socket is unsupported")
            }
        }

        interp_ok(Scalar::from_i32(0))
    }

    /// For more information on the arguments see the accept manpage:
    /// <https://linux.die.net/man/2/accept4>
    fn accept4(
        &mut self,
        socket: &OpTy<'tcx>,
        address: &OpTy<'tcx>,
        address_len: &OpTy<'tcx>,
        flags: Option<&OpTy<'tcx>>,
        // Location where the output scalar is written to.
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let address_ptr = this.read_pointer(address)?;
        let address_len_ptr = this.read_pointer(address_len)?;
        let mut flags =
            if let Some(flags) = flags { this.read_scalar(flags)?.to_i32()? } else { 0 };

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return(LibcError("EBADF"), dest);
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return this.set_last_error_and_return(LibcError("ENOTSOCK"), dest);
        };

        assert!(this.machine.communicate(), "cannot have `Socket` with isolation enabled!");

        if !matches!(*socket.state.borrow(), SocketState::Listening(_)) {
            throw_unsup_format!(
                "accept4: accepting incoming connections is only allowed when socket is listening"
            )
        };

        let mut is_client_sock_nonblock = false;

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
                is_client_sock_nonblock = true;
                flags &= !sock_nonblock;
            }
            if flags & sock_cloexec == sock_cloexec {
                // We don't support `exec` so we can ignore this.
                flags &= !sock_cloexec;
            }
        }

        if flags != 0 {
            throw_unsup_format!(
                "accept4: flag {flags:#x} is unsupported, only SOCK_CLOEXEC \
                and SOCK_NONBLOCK are allowed",
            );
        }

        if socket.is_non_block.get() {
            // We have a non-blocking socket and thus don't want to block until
            // we can accept an incoming connection.
            match this.try_non_block_accept(
                &socket,
                address_ptr,
                address_len_ptr,
                is_client_sock_nonblock,
            )? {
                Ok(sockfd) => {
                    // We need to create the scalar using the destination size since
                    // `syscall(SYS_accept4, ...)` returns a long which doesn't match
                    // the int returned from the `accept`/`accept4` syscalls.
                    // See <https://man7.org/linux/man-pages/man2/syscall.2.html>.
                    this.write_scalar(Scalar::from_int(sockfd, dest.layout.size), dest)
                }
                Err(e) => this.set_last_error_and_return(e, dest),
            }
        } else {
            // The socket is in blocking mode and thus the accept call should block
            // until an incoming connection is ready.
            this.block_for_accept(
                socket,
                address_ptr,
                address_len_ptr,
                is_client_sock_nonblock,
                dest.clone(),
            )
        }
    }

    fn connect(
        &mut self,
        socket: &OpTy<'tcx>,
        address: &OpTy<'tcx>,
        address_len: &OpTy<'tcx>,
        // Location where the output scalar is written to.
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let address = match this.read_socket_address(address, address_len, "connect")? {
            Ok(address) => address,
            Err(e) => return this.set_last_error_and_return(e, dest),
        };

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return(LibcError("EBADF"), dest);
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket
            return this.set_last_error_and_return(LibcError("ENOTSOCK"), dest);
        };

        assert!(this.machine.communicate(), "cannot have `Socket` with isolation enabled!");

        match &*socket.state.borrow() {
            SocketState::Initial => { /* fall-through to below */ }
            // The socket is already in a connecting state.
            SocketState::Connecting(_) =>
                return this.set_last_error_and_return(LibcError("EALREADY"), dest),
            // We don't return EISCONN for already connected sockets, for which we're
            // sure that the connection is established, since TCP sockets are usually
            // allowed to be connected multiple times.
            _ =>
                throw_unsup_format!(
                    "connect: connecting is only supported for sockets which are neither \
                    bound, listening nor already connected"
                ),
        }

        // This begins establishing the connection, but does not block until the stream is fully connected.
        // We deal with that below.
        match TcpStream::connect(address) {
            Ok(stream) => {
                *socket.state.borrow_mut() = SocketState::Connecting(stream);
                // Register the socket to the blocking I/O manager because
                // we now have an associated host socket.
                this.machine.blocking_io.register(socket.clone());
            }
            Err(e) => return this.set_last_error_and_return(e, dest),
        };

        if socket.is_non_block.get() {
            // We have a non-blocking socket and thus don't want to block until
            // the connection is established.

            // Since the [`TcpStream::connect`] function of mio hides the EINPROGRESS
            // we just always return EINPROGRESS and check whether the connection succeeded
            // once we want to use the connected socket.
            this.set_last_error_and_return(LibcError("EINPROGRESS"), dest)
        } else {
            // The socket is in blocking mode and thus the connect call should block
            // until the connection with the server is established.

            let dest = dest.clone();

            this.ensure_connected(
                socket,
                /* should_wait */ true,
                "connect",
                callback!(
                    @capture<'tcx> {
                        dest: MPlaceTy<'tcx>
                    } |this, result: Result<(), ()>| {
                        if result.is_err() {
                            this.set_last_error_and_return(LibcError("ENOTCONN"), &dest)
                        } else {
                            this.write_scalar(Scalar::from_i32(0), &dest)
                        }
                    }
                ),
            )
        }
    }

    fn send(
        &mut self,
        socket: &OpTy<'tcx>,
        buffer: &OpTy<'tcx>,
        length: &OpTy<'tcx>,
        flags: &OpTy<'tcx>,
        // Location where the output scalar is written to.
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let buffer_ptr = this.read_pointer(buffer)?;
        let size_layout = this.libc_ty_layout("size_t");
        let length: usize =
            this.read_scalar(length)?.to_uint(size_layout.size)?.try_into().unwrap();
        let mut flags = this.read_scalar(flags)?.to_i32()?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return(LibcError("EBADF"), dest);
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket
            return this.set_last_error_and_return(LibcError("ENOTSOCK"), dest);
        };

        let mut is_op_non_block = false;

        // Interpret the flag. Every flag we recognize is "subtracted" from `flags`, so
        // if there is anything left at the end, that's an unsupported flag.
        if matches!(
            this.tcx.sess.target.os,
            Os::Linux | Os::Android | Os::FreeBsd | Os::Solaris | Os::Illumos
        ) {
            // MSG_NOSIGNAL and MSG_DONTWAIT only exist on Linux, Android, FreeBSD,
            // Solaris, and Illumos targets.
            let msg_nosignal = this.eval_libc_i32("MSG_NOSIGNAL");
            let msg_dontwait = this.eval_libc_i32("MSG_DONTWAIT");
            if flags & msg_nosignal == msg_nosignal {
                // This is only needed to ensure that no EPIPE signal is sent when
                // trying to send into a stream which is no longer connected.
                // Since we don't support signals, we can ignore this.
                flags &= !msg_nosignal;
            }
            if flags & msg_dontwait == msg_dontwait {
                flags &= !msg_dontwait;
                is_op_non_block = true;
            }
        }

        if flags != 0 {
            throw_unsup_format!(
                "send: flag {flags:#x} is unsupported, only MSG_NOSIGNAL and MSG_DONTWAIT are allowed",
            );
        }

        // If either the operation or the socket is non-blocking, we don't want
        // to wait until the connection is established.
        let should_wait = !is_op_non_block && !socket.is_non_block.get();
        let dest = dest.clone();

        this.ensure_connected(
            socket.clone(),
            should_wait,
            "send",
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<Socket>,
                    flags: i32,
                    buffer_ptr: Pointer,
                    length: usize,
                    is_op_non_block: bool,
                    dest: MPlaceTy<'tcx>,
                } |this, result: Result<(), ()>| {
                    if result.is_err() {
                        return this.set_last_error_and_return(LibcError("ENOTCONN"), &dest)
                    }

                    if is_op_non_block || socket.is_non_block.get() {
                        // We have a non-blocking operation or a non-blocking socket and
                        // thus don't want to block until we can send.
                        match this.try_non_block_send(&socket, buffer_ptr, length)? {
                            Ok(size) => this.write_scalar(Scalar::from_target_isize(size.try_into().unwrap(), this), &dest),
                            Err(e) => this.set_last_error_and_return(e, &dest),
                        }
                    } else {
                        // The socket is in blocking mode and thus the send call should block
                        // until we can send some bytes into the socket.
                        this.block_for_send(
                            socket,
                            buffer_ptr,
                            length,
                            callback!(@capture<'tcx> {
                                dest: MPlaceTy<'tcx>
                            } |this, result: Result<usize, IoError>| {
                                match result {
                                    Ok(size) => this.write_scalar(Scalar::from_target_isize(size.try_into().unwrap(), this), &dest),
                                    Err(e) => this.set_last_error_and_return(e, &dest)
                                }
                            }),
                        )
                    }
                }
            ),
        )
    }

    fn recv(
        &mut self,
        socket: &OpTy<'tcx>,
        buffer: &OpTy<'tcx>,
        length: &OpTy<'tcx>,
        flags: &OpTy<'tcx>,
        // Location where the output scalar is written to.
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let buffer_ptr = this.read_pointer(buffer)?;
        let size_layout = this.libc_ty_layout("size_t");
        let length: usize =
            this.read_scalar(length)?.to_uint(size_layout.size)?.try_into().unwrap();
        let mut flags = this.read_scalar(flags)?.to_i32()?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return(LibcError("EBADF"), dest);
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket
            return this.set_last_error_and_return(LibcError("ENOTSOCK"), dest);
        };

        let mut should_peek = false;
        let mut is_op_non_block = false;

        // Interpret the flag. Every flag we recognize is "subtracted" from `flags`, so
        // if there is anything left at the end, that's an unsupported flag.

        let msg_peek = this.eval_libc_i32("MSG_PEEK");
        if flags & msg_peek == msg_peek {
            should_peek = true;
            flags &= !msg_peek;
        }

        if matches!(this.tcx.sess.target.os, Os::Linux | Os::Android | Os::FreeBsd | Os::Illumos) {
            // MSG_CMSG_CLOEXEC only exists on Linux, Android, FreeBSD,
            // and Illumos targets.
            let msg_cmsg_cloexec = this.eval_libc_i32("MSG_CMSG_CLOEXEC");
            if flags & msg_cmsg_cloexec == msg_cmsg_cloexec {
                // We don't support `exec` so we can ignore this.
                flags &= !msg_cmsg_cloexec;
            }
        }

        if matches!(
            this.tcx.sess.target.os,
            Os::Linux | Os::Android | Os::FreeBsd | Os::Solaris | Os::Illumos
        ) {
            // MSG_DONTWAIT only exists on Linux, Android, FreeBSD,
            // Solaris, and Illumos targets.
            let msg_dontwait = this.eval_libc_i32("MSG_DONTWAIT");
            if flags & msg_dontwait == msg_dontwait {
                flags &= !msg_dontwait;
                is_op_non_block = true;
            }
        }

        if flags != 0 {
            throw_unsup_format!(
                "recv: flag {flags:#x} is unsupported, only MSG_PEEK, MSG_DONTWAIT \
                and MSG_CMSG_CLOEXEC are allowed",
            );
        }

        // If either the operation or the socket is non-blocking, we don't want
        // to wait until the connection is established.
        let should_wait = !is_op_non_block && !socket.is_non_block.get();
        let dest = dest.clone();

        this.ensure_connected(
            socket.clone(),
            should_wait,
            "recv",
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<Socket>,
                    buffer_ptr: Pointer,
                    length: usize,
                    should_peek: bool,
                    is_op_non_block: bool,
                    dest: MPlaceTy<'tcx>,
                } |this, result: Result<(), ()>| {
                    if result.is_err() {
                        return this.set_last_error_and_return(LibcError("ENOTCONN"), &dest)
                    }

                    if is_op_non_block || socket.is_non_block.get() {
                        // We have a non-blocking operation or a non-blocking socket and
                        // thus don't want to block until we can receive.
                        match this.try_non_block_recv(&socket, buffer_ptr, length, should_peek)? {
                            Ok(size) => this.write_scalar(Scalar::from_target_isize(size.try_into().unwrap(), this), &dest),
                            Err(e) => this.set_last_error_and_return(e, &dest),
                        }
                    } else {
                        // The socket is in blocking mode and thus the receive call should block
                        // until we can receive some bytes from the socket.
                        this.block_for_recv(
                            socket,
                            buffer_ptr,
                            length,
                            should_peek,
                            callback!(@capture<'tcx> {
                                dest: MPlaceTy<'tcx>
                            } |this, result: Result<usize, IoError>| {
                                match result {
                                    Ok(size) => this.write_scalar(Scalar::from_target_isize(size.try_into().unwrap(), this), &dest),
                                    Err(e) => this.set_last_error_and_return(e, &dest)
                                }
                            }),
                        )
                    }
                }
            ),
        )
    }

    fn setsockopt(
        &mut self,
        socket: &OpTy<'tcx>,
        level: &OpTy<'tcx>,
        option_name: &OpTy<'tcx>,
        option_value: &OpTy<'tcx>,
        option_len: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let level = this.read_scalar(level)?.to_i32()?;
        let option_name = this.read_scalar(option_name)?.to_i32()?;
        let socklen_layout = this.libc_ty_layout("socklen_t");
        let option_len = this.read_scalar(option_len)?.to_int(socklen_layout.size)?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };

        let Some(_socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return this.set_last_error_and_return_i32(LibcError("ENOTSOCK"));
        };

        if level == this.eval_libc_i32("SOL_SOCKET") {
            let opt_so_reuseaddr = this.eval_libc_i32("SO_REUSEADDR");

            if matches!(this.tcx.sess.target.os, Os::MacOs | Os::FreeBsd | Os::NetBsd) {
                // SO_NOSIGPIPE only exists on MacOS, FreeBSD, and NetBSD.
                let opt_so_nosigpipe = this.eval_libc_i32("SO_NOSIGPIPE");

                if option_name == opt_so_nosigpipe {
                    if option_len != 4 {
                        // Option value should be C-int which is usually 4 bytes.
                        return this.set_last_error_and_return_i32(LibcError("EINVAL"));
                    }
                    let option_value =
                        this.deref_pointer_as(option_value, this.machine.layouts.i32)?;
                    let _val = this.read_scalar(&option_value)?.to_i32()?;
                    // We entirely ignore this value since we do not support signals anyway.

                    return interp_ok(Scalar::from_i32(0));
                }
            }

            if option_name == opt_so_reuseaddr {
                if option_len != 4 {
                    // Option value should be C-int which is usually 4 bytes.
                    return this.set_last_error_and_return_i32(LibcError("EINVAL"));
                }
                let option_value = this.deref_pointer_as(option_value, this.machine.layouts.i32)?;
                let _val = this.read_scalar(&option_value)?.to_i32()?;
                // We entirely ignore this: std always sets REUSEADDR for us, and in the end it's more of a
                // hint to bypass some arbitrary timeout anyway.
                return interp_ok(Scalar::from_i32(0));
            } else {
                throw_unsup_format!(
                    "setsockopt: option {option_name:#x} is unsupported for level SOL_SOCKET",
                );
            }
        }

        throw_unsup_format!(
            "setsockopt: level {level:#x} is unsupported, only SOL_SOCKET is allowed"
        );
    }

    fn getsockopt(
        &mut self,
        socket: &OpTy<'tcx>,
        level: &OpTy<'tcx>,
        option_name: &OpTy<'tcx>,
        option_value: &OpTy<'tcx>,
        option_len: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let level = this.read_scalar(level)?.to_i32()?;
        let option_name = this.read_scalar(option_name)?.to_i32()?;
        // These two pointers are used to return the value: `len_ptr` initially stores how much space
        // is available. If the actual value fits into that space, it is written to
        // `value_ptr` and `len_ptr` is updated to represent how many bytes
        // were actually written. If the value does not fit, it is silently truncated.
        // Also see <https://pubs.opengroup.org/onlinepubs/9799919799/functions/getsockopt.html>.
        let option_value_ptr = this.read_pointer(option_value)?;
        let option_len_ptr = this.read_pointer(option_len)?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return this.set_last_error_and_return_i32(LibcError("ENOTSOCK"));
        };

        if option_value_ptr == Pointer::null() || option_len_ptr == Pointer::null() {
            // This socket option returns a value and thus we need to return EFAULT
            // when either the value or the length pointers are null pointers.
            return this.set_last_error_and_return_i32(LibcError("EFAULT"));
        }

        let socklen_layout = this.libc_ty_layout("socklen_t");
        let option_len_ptr_mplace = this.ptr_to_mplace(option_len_ptr, socklen_layout);
        let option_len: usize = this
            .read_scalar(&option_len_ptr_mplace)?
            .to_int(socklen_layout.size)?
            .try_into()
            .unwrap();

        // We need a temporary buffer as `option_value_ptr` might not point to a large enough
        // buffer, in which case we have to truncate.
        let value_buffer = if level == this.eval_libc_i32("SOL_SOCKET") {
            let opt_so_error = this.eval_libc_i32("SO_ERROR");

            if option_name == opt_so_error {
                // Because `TcpStream::take_error()` and `TcpListener::take_error()` consume the latest async
                // error, we know that our stored `socket.error` is outdated when `TcpStream::take_error()`/
                // `TcpListener::take_error()` returns `Ok(Some(...))`.
                // If they return `Ok(None)`, then we fall back to the stored `socket.error`.
                let error = match &*socket.state.borrow() {
                    SocketState::Initial | SocketState::Bound(_) => socket.error.take(),
                    SocketState::Listening(listener) =>
                        listener.take_error().unwrap_or(socket.error.take()),
                    SocketState::Connecting(stream) | SocketState::Connected(stream) =>
                        stream.take_error().unwrap_or(socket.error.take()),
                };
                // Clear our own stored error -- it was either `take`n above or it is outdated.
                socket.error.replace(None);

                // We know there is no longer an async error and thus we need to update the
                // I/O and epoll readiness of the socket.
                socket.io_readiness.borrow_mut().error = false;
                this.update_epoll_active_events(socket, /* force_edge */ false)?;

                let return_value = match error {
                    Some(err) => this.io_error_to_errnum(err)?.to_i32()?,
                    // If there is no error, we write 0 into the option value buffer.
                    None => 0,
                };

                // Allocate new buffer on the stack with the `i32` layout.
                let value_buffer = this.allocate(this.machine.layouts.i32, MemoryKind::Stack)?;
                this.write_int(return_value, &value_buffer)?;
                value_buffer
            } else {
                throw_unsup_format!(
                    "getsockopt: option {option_name:#x} is unsupported for level SOL_SOCKET",
                );
            }
        } else if level == this.eval_libc_i32("IPPROTO_IP") {
            let opt_ip_ttl = this.eval_libc_i32("IP_TTL");

            if option_name == opt_ip_ttl {
                let ttl = match &*socket.state.borrow() {
                    SocketState::Initial | SocketState::Bound(_) =>
                        throw_unsup_format!(
                            "getsockopt: reading option IP_TTL on level IPPROTO_IP is only supported \
                            on connected and listening sockets"
                        ),
                    SocketState::Listening(listener) => listener.ttl(),
                    SocketState::Connecting(stream) | SocketState::Connected(stream) =>
                        stream.ttl(),
                };

                let ttl = match ttl {
                    Ok(ttl) => ttl,
                    Err(e) => return this.set_last_error_and_return_i32(e),
                };

                // Allocate new buffer on the stack with the `u32` layout.
                let value_buffer = this.allocate(this.machine.layouts.u32, MemoryKind::Stack)?;
                this.write_int(ttl, &value_buffer)?;
                value_buffer
            } else {
                throw_unsup_format!(
                    "getsockopt: option {option_name:#x} is unsupported for level IPPROTO_IP",
                );
            }
        } else {
            throw_unsup_format!(
                "getsockopt: level {level:#x} is unsupported, only SOL_SOCKET is allowed"
            )
        };

        // Truncated size of the output value.
        let output_value_len = value_buffer.layout.size.min(Size::from_bytes(option_len));
        // Copy the truncated value into the buffer pointed to by `option_value_ptr`.
        this.mem_copy(
            value_buffer.ptr(),
            option_value_ptr,
            // Truncate the value to fit the provided buffer.
            output_value_len,
            // The buffers are guaranteed to not overlap since the `value_buffer`
            // was just newly allocated on the stack.
            true,
        )?;
        // Deallocate the value buffer as it was only needed to store the value and
        // copy it into the buffer pointed to by `option_value_ptr`.
        this.deallocate_ptr(value_buffer.ptr(), None, MemoryKind::Stack)?;

        // On output, the length pointer contains the amount of bytes written -- not the size
        // of the value before truncation.
        this.write_scalar(
            Scalar::from_uint(output_value_len.bytes(), socklen_layout.size),
            &option_len_ptr_mplace,
        )?;

        interp_ok(Scalar::from_i32(0))
    }

    fn getsockname(
        &mut self,
        socket: &OpTy<'tcx>,
        address: &OpTy<'tcx>,
        address_len: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let address_ptr = this.read_pointer(address)?;
        let address_len_ptr = this.read_pointer(address_len)?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return this.set_last_error_and_return_i32(LibcError("ENOTSOCK"));
        };

        assert!(this.machine.communicate(), "cannot have `Socket` with isolation enabled!");

        let state = socket.state.borrow();

        let address = match &*state {
            SocketState::Bound(address) => {
                if address.port() == 0 {
                    // The socket is bound to a zero-port which means it gets assigned a random
                    // port. Since we don't yet have an underlying socket, we don't know what this
                    // random port will be and thus this is unsupported.
                    throw_unsup_format!(
                        "getsockname: when the port is 0, getting the socket address before \
                        calling `listen` or `connect` is unsupported"
                    )
                }

                *address
            }
            SocketState::Listening(listener) =>
                match listener.local_addr() {
                    Ok(address) => address,
                    Err(e) => return this.set_last_error_and_return_i32(e),
                },
            SocketState::Connecting(stream) | SocketState::Connected(stream) => {
                if cfg!(windows) && matches!(&*state, SocketState::Connecting(_)) {
                    // FIXME: On Windows hosts `TcpStream::local_addr` returns `0.0.0.0:0` whilst
                    // the socket is connecting:
                    // <https://learn.microsoft.com/en-us/windows/win32/api/winsock/nf-winsock-getsockname#remarks>
                    // This is problematic because UNIX targets could expect a real local address even
                    // for a connecting non-blocking socket.

                    static DEDUP: AtomicBool = AtomicBool::new(false);
                    if !DEDUP.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        this.emit_diagnostic(NonHaltingDiagnostic::ConnectingSocketGetsockname);
                    }
                }
                match stream.local_addr() {
                    Ok(address) => address,
                    Err(e) => return this.set_last_error_and_return_i32(e),
                }
            }
            // For non-bound sockets the POSIX manual says the returned address is unspecified.
            // Often this is 0.0.0.0:0 and thus we set it to this value.
            SocketState::Initial => SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)),
        };

        this.write_socket_address(&address, address_ptr, address_len_ptr, "getsockname")
            .map(|_| Scalar::from_i32(0))
    }

    fn getpeername(
        &mut self,
        socket: &OpTy<'tcx>,
        address: &OpTy<'tcx>,
        address_len: &OpTy<'tcx>,
        // Location where the output scalar is written to.
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let address_ptr = this.read_pointer(address)?;
        let address_len_ptr = this.read_pointer(address_len)?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return(LibcError("EBADF"), dest);
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return this.set_last_error_and_return(LibcError("ENOTSOCK"), dest);
        };

        assert!(this.machine.communicate(), "cannot have `Socket` with isolation enabled!");

        let dest = dest.clone();

        // It's only safe to call [`TcpStream::peer_addr`] after the socket is connected since
        // UNIX targets should return ENOTCONN when the connection is not yet established.
        this.ensure_connected(
            socket.clone(),
            /* should_wait */ false,
            "getpeername",
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<Socket>,
                    address_ptr: Pointer,
                    address_len_ptr: Pointer,
                    dest: MPlaceTy<'tcx>,
                } |this, result: Result<(), ()>| {
                    if result.is_err() {
                        return this.set_last_error_and_return(LibcError("ENOTCONN"), &dest)
                    };

                    let SocketState::Connected(stream) = &*socket.state.borrow() else {
                        unreachable!()
                    };

                    let address = match stream.peer_addr() {
                        Ok(address) => address,
                        Err(e) => return this.set_last_error_and_return(e, &dest),
                    };

                    this.write_socket_address(
                        &address,
                        address_ptr,
                        address_len_ptr,
                        "getpeername",
                    )?;
                   this.write_scalar(Scalar::from_i32(0), &dest)
                }
            ),
        )
    }

    fn shutdown(&mut self, socket: &OpTy<'tcx>, how: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let how = this.read_scalar(how)?.to_i32()?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return this.set_last_error_and_return_i32(LibcError("ENOTSOCK"));
        };

        assert!(this.machine.communicate(), "cannot have `Socket` with isolation enabled!");

        let state = socket.state.borrow();

        let (SocketState::Connecting(stream) | SocketState::Connected(stream)) = &*state else {
            return this.set_last_error_and_return_i32(LibcError("ENOTCONN"));
        };

        let is_read_shutdown = how == this.eval_libc_i32("SHUT_RD");
        let is_write_shutdown = how == this.eval_libc_i32("SHUT_WR");
        let is_read_write_shutdown = how == this.eval_libc_i32("SHUT_RDWR");

        let how = match () {
            _ if is_read_shutdown => Shutdown::Read,
            _ if is_write_shutdown => Shutdown::Write,
            _ if is_read_write_shutdown => Shutdown::Both,
            // An invalid value was passed to `how`.
            _ => return this.set_last_error_and_return_i32(LibcError("EINVAL")),
        };

        if let Err(e) = stream.shutdown(how) {
            return this.set_last_error_and_return_i32(e);
        };

        drop(state);

        // Because we map cross platform mio readiness to epoll readiness and
        // the different platforms don't treat `shutdown` the same way, we set
        // the readiness after a `shutdown` manually to achieve more consistent
        // epoll readiness. Otherwise we do not generate enough epoll events
        // on partial shutdowns on Windows hosts.
        let mut readiness = socket.io_readiness.borrow_mut();
        // Closing the read end of a socket causes an EPOLLRDHUP event.
        readiness.read_closed |= is_read_shutdown || is_read_write_shutdown;
        // Only shutting down the write end doesn't cause an EPOLLHUP event
        // and thus we won't set the `write_closed` readiness for it here.
        readiness.write_closed |= is_read_write_shutdown;
        // The Linux kernel also sets EPOLLIN when both ends of a socket are closed:
        // <https://github.com/torvalds/linux/blob/HEAD/net/ipv4/tcp.c#L584-L588>
        readiness.readable |= is_read_write_shutdown;

        drop(readiness);

        // Update the epoll readiness for the socket.
        this.update_epoll_active_events(socket, /* force_edge */ false)?;

        interp_ok(Scalar::from_i32(0))
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Block the thread until there's an incoming connection or an error occurred.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    ///
    /// **Note**: This function is only safe to call when having previously ensured
    /// that the socket is in [`SocketState::Listening`].
    fn block_for_accept(
        &mut self,
        socket: FileDescriptionRef<Socket>,
        address_ptr: Pointer,
        address_len_ptr: Pointer,
        is_client_sock_nonblock: bool,
        dest: MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Read,
            None,
            callback!(@capture<'tcx> {
                address_ptr: Pointer,
                address_len_ptr: Pointer,
                is_client_sock_nonblock: bool,
                socket: FileDescriptionRef<Socket>,
                dest: MPlaceTy<'tcx>,
            } |this, kind: UnblockKind| {
                assert_eq!(kind, UnblockKind::Ready);

                // Remove the blocking I/O interest for unblocking this thread.
                this.machine.blocking_io.remove_blocked_thread(socket.id(), this.machine.threads.active_thread());

                match this.try_non_block_accept(&socket, address_ptr, address_len_ptr, is_client_sock_nonblock)? {
                    Ok(sockfd) => {
                        // We need to create the scalar using the destination size since
                        // `syscall(SYS_accept4, ...)` returns a long which doesn't match
                        // the int returned from the `accept`/`accept4` syscalls.
                        // See <https://man7.org/linux/man-pages/man2/syscall.2.html>.
                        this.write_scalar(Scalar::from_int(sockfd, dest.layout.size), &dest)
                    },
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        this.block_for_accept(socket, address_ptr, address_len_ptr, is_client_sock_nonblock, dest)
                    }
                    Err(e) => this.set_last_error_and_return(e, &dest),
                }
            }),
        )
    }

    /// Attempt to accept an incoming connection on the listening socket in a
    /// non-blocking manner.
    ///
    /// **Note**: This function is only safe to call when having previously ensured
    /// that the socket is in [`SocketState::Listening`].
    fn try_non_block_accept(
        &mut self,
        socket: &FileDescriptionRef<Socket>,
        address_ptr: Pointer,
        address_len_ptr: Pointer,
        is_client_sock_nonblock: bool,
    ) -> InterpResult<'tcx, Result<i32, IoError>> {
        let this = self.eval_context_mut();

        let state = socket.state.borrow();
        let SocketState::Listening(listener) = &*state else {
            panic!(
                "try_non_block_accept must only be called when socket is in `SocketState::Listening`"
            )
        };

        let (stream, addr) = match listener.accept() {
            Ok(peer) => peer,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // We know that the source is not readable so we need to update its readiness.
                socket.io_readiness.borrow_mut().readable = false;
                this.update_epoll_active_events(socket.clone(), /* force_edge */ false)?;

                return interp_ok(Err(IoError::HostError(e)));
            }
            Err(e) => return interp_ok(Err(IoError::HostError(e))),
        };

        let family = match addr {
            SocketAddr::V4(_) => SocketFamily::IPv4,
            SocketAddr::V6(_) => SocketFamily::IPv6,
        };

        if address_ptr != Pointer::null() {
            // We only attempt a write if the address pointer is not a null pointer.
            // If the address pointer is a null pointer the user isn't interested in the
            // address and we don't need to write anything.
            this.write_socket_address(&addr, address_ptr, address_len_ptr, "accept4")?;
        }

        let fd = this.machine.fds.new_ref(Socket {
            family,
            state: RefCell::new(SocketState::Connected(stream)),
            is_non_block: Cell::new(is_client_sock_nonblock),
            io_readiness: RefCell::new(BlockingIoSourceReadiness::empty()),
            error: RefCell::new(None),
        });
        // Register the socket to the blocking I/O manager because
        // there is an associated host socket.
        this.machine.blocking_io.register(fd.clone());
        let sockfd = this.machine.fds.insert(fd);
        interp_ok(Ok(sockfd))
    }

    /// Block the thread until we can send bytes into the connected socket
    /// or an error occurred.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    ///
    /// **Note**: This function is only safe to call when having previously ensured
    /// that the socket is in [`SocketState::Connected`].
    fn block_for_send(
        &mut self,
        socket: FileDescriptionRef<Socket>,
        buffer_ptr: Pointer,
        length: usize,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Write,
            None,
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<Socket>,
                buffer_ptr: Pointer,
                length: usize,
                finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
            } |this, kind: UnblockKind| {
                assert_eq!(kind, UnblockKind::Ready);

                // Remove the blocking I/O interest for unblocking this thread.
                this.machine.blocking_io.remove_blocked_thread(socket.id(), this.machine.threads.active_thread());

                match this.try_non_block_send(&socket, buffer_ptr, length)? {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        this.block_for_send(socket, buffer_ptr, length, finish)
                    },
                    result => finish.call(this, result)
                }
            }),
        )
    }

    /// Attempt to send bytes into the connected socket in a non-blocking manner.
    ///
    /// **Note**: This function is only safe to call when having previously ensured
    /// that the socket is in [`SocketState::Connected`].
    fn try_non_block_send(
        &mut self,
        socket: &FileDescriptionRef<Socket>,
        buffer_ptr: Pointer,
        length: usize,
    ) -> InterpResult<'tcx, Result<usize, IoError>> {
        let this = self.eval_context_mut();

        let SocketState::Connected(stream) = &mut *socket.state.borrow_mut() else {
            panic!("try_non_block_send must only be called when the socket is connected")
        };

        // This is a *non-blocking* write.
        let result = this.write_to_host(stream, length, buffer_ptr)?;
        match result {
            Err(IoError::HostError(e))
                if matches!(e.kind(), io::ErrorKind::NotConnected | io::ErrorKind::WouldBlock) =>
            {
                // We know that the source is not writable so we need to update it's readiness.
                socket.io_readiness.borrow_mut().writable = false;
                this.update_epoll_active_events(socket.clone(), /* force_edge */ false)?;

                // On Windows hosts, `send` can return WSAENOTCONN where EAGAIN or EWOULDBLOCK
                // would be returned on UNIX-like systems. We thus remap this error to an EWOULDBLOCK.
                interp_ok(Err(IoError::HostError(io::ErrorKind::WouldBlock.into())))
            }
            Ok(bytes_written) if bytes_written < length => {
                // We had a short write. On Unix hosts using the `epoll` and `kqueue` backends, a
                // short write means that the write buffer is full. We update the readiness
                // accordingly, which means that next time we see "writable" we will report an epoll
                // edge. Some applications (e.g. tokio) rely on this behavior; see
                // <https://github.com/tokio-rs/tokio/blob/HEAD/tokio/src/io/poll_evented.rs#L244-L264>.
                if cfg!(any(
                    // epoll
                    target_os = "android",
                    target_os = "illumos",
                    target_os = "linux",
                    target_os = "redox",
                    // kqueue
                    target_os = "dragonfly",
                    target_os = "freebsd",
                    target_os = "ios",
                    target_os = "macos",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "tvos",
                    target_os = "visionos",
                    target_os = "watchos",
                )) {
                    socket.io_readiness.borrow_mut().writable = false;
                    this.update_epoll_active_events(socket.clone(), /* force_edge */ false)?;
                } else {
                    // On hosts which don't use the `epoll` or `kqueue` backends, a short write
                    // doesn't imply a full write buffer. However, the target we are emulating might
                    // guarantee this behavior. To prevent applications from being stuck on such
                    // targets waiting on a new readiness event, we emit a new edge which still
                    // contains a writable readiness. This should trick the applications into trying
                    // another write which would then return EWOULDBLOCK should it really be full.
                    // This results in an unrealistic execution but we don't have another way of
                    // finding out whether the write buffer is full. The "default case" of linux
                    // host and linux target isn't affected by this.
                    this.update_epoll_active_events(socket.clone(), /* force_edge */ true)?;
                }
                interp_ok(result)
            }
            result => interp_ok(result),
        }
    }

    /// Block the thread until we can receive bytes from the connected socket
    /// or an error occurred.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    ///
    /// **Note**: This function is only safe to call when having previously ensured
    /// that the socket is in [`SocketState::Connected`].
    fn block_for_recv(
        &mut self,
        socket: FileDescriptionRef<Socket>,
        buffer_ptr: Pointer,
        length: usize,
        should_peek: bool,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Read,
            None,
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<Socket>,
                buffer_ptr: Pointer,
                length: usize,
                should_peek: bool,
                finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
            } |this, kind: UnblockKind| {
                assert_eq!(kind, UnblockKind::Ready);

                // Remove the blocking I/O interest for unblocking this thread.
                this.machine.blocking_io.remove_blocked_thread(socket.id(), this.machine.threads.active_thread());

                match this.try_non_block_recv(&socket, buffer_ptr, length, should_peek)? {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        this.block_for_recv(socket, buffer_ptr, length, should_peek, finish)
                    },
                    result => finish.call(this, result)
                }
            }),
        )
    }

    /// Attempt to receive bytes from the connected socket in a non-blocking manner.
    ///
    /// **Note**: This function is only safe to call when having previously ensured
    /// that the socket is in [`SocketState::Connected`].
    fn try_non_block_recv(
        &mut self,
        socket: &FileDescriptionRef<Socket>,
        buffer_ptr: Pointer,
        length: usize,
        should_peek: bool,
    ) -> InterpResult<'tcx, Result<usize, IoError>> {
        let this = self.eval_context_mut();

        let SocketState::Connected(stream) = &mut *socket.state.borrow_mut() else {
            panic!("try_non_block_recv must only be called when the socket is connected")
        };

        // This is a *non-blocking* read/peek.
        let result = this.read_from_host(
            |buf| {
                if should_peek { stream.peek(buf) } else { stream.read(buf) }
            },
            length,
            buffer_ptr,
        )?;
        match result {
            Err(IoError::HostError(e))
                if matches!(e.kind(), io::ErrorKind::NotConnected | io::ErrorKind::WouldBlock) =>
            {
                // We know that the source is not readable so we need to update it's readiness.
                socket.io_readiness.borrow_mut().readable = false;
                this.update_epoll_active_events(socket.clone(), /* force_edge */ false)?;

                // On Windows hosts, `recv` can return WSAENOTCONN where EAGAIN or EWOULDBLOCK
                // would be returned on UNIX-like systems. We thus remap this error to an EWOULDBLOCK.
                interp_ok(Err(IoError::HostError(io::ErrorKind::WouldBlock.into())))
            }
            Ok(bytes_read) if bytes_read < length && bytes_read > 0 => {
                // We had a short read. (Note that reading 0 bytes is guaranteed to indicate EOF,
                // and can never happen spuriously, so we have to exclude that case.) On Unix hosts
                // using the `epoll` and `kqueue` backends, a short read means that the read buffer
                // is empty. We update the readiness accordingly, which means that next time we see
                // "readable" we will report an epoll edge. Some applications (e.g. tokio) rely on
                // this behavior; see
                // <https://github.com/tokio-rs/tokio/blob/HEAD/tokio/src/io/poll_evented.rs#L190-L210>
                if cfg!(any(
                    // epoll
                    target_os = "android",
                    target_os = "illumos",
                    target_os = "linux",
                    target_os = "redox",
                    // kqueue
                    target_os = "dragonfly",
                    target_os = "freebsd",
                    target_os = "ios",
                    target_os = "macos",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "tvos",
                    target_os = "visionos",
                    target_os = "watchos",
                )) {
                    socket.io_readiness.borrow_mut().readable = false;
                    this.update_epoll_active_events(socket.clone(), /* force_edge */ false)?;
                } else {
                    // On hosts which don't use the `epoll` or `kqueue` backends, a short read
                    // doesn't imply an empty read buffer. However, the target we are emulating
                    // might guarantee this behavior. To prevent applications from being stuck on
                    // such targets waiting on a new readiness event, we emit a new edge which still
                    // contains a readable readiness. This should trick the applications into trying
                    // another read which would then return EWOULDBLOCK should it really be empty.
                    // This results in an unrealistic execution but we don't have another way of
                    // finding out whether the read buffer is empty. The "default case" of linux
                    // host and linux target isn't affected by this.
                    this.update_epoll_active_events(socket.clone(), /* force_edge */ true)?;
                }
                interp_ok(result)
            }
            result => interp_ok(result),
        }
    }

    // Execute the provided callback function when the socket is either in
    // [`SocketState::Connected`] or an error occurred.
    /// If the socket is currently neither in the [`SocketState::Connecting`] nor
    /// the [`SocketState::Connecting`] state, an ENOTCONN error is returned.
    /// When the callback function is called with `Ok(_)`, then we're guaranteed
    /// that the socket is in the [`SocketState::Connected`] state.
    ///
    /// This function can optionally also block until either an error occurred or
    /// the socket reached the [`SocketState::Connected`] state.
    fn ensure_connected(
        &mut self,
        socket: FileDescriptionRef<Socket>,
        should_wait: bool,
        foreign_name: &'static str,
        action: DynMachineCallback<'tcx, Result<(), ()>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let state = socket.state.borrow();
        match &*state {
            SocketState::Connecting(_) => { /* fall-through to below */ }
            SocketState::Connected(_) => {
                drop(state);
                return action.call(this, Ok(()));
            }
            _ => {
                drop(state);
                return action.call(this, Err(()));
            }
        };

        drop(state);

        // We're currently connecting. Since the underlying mio socket is non-blocking,
        // the only way to determine whether we are done connecting is by polling.
        // If we should wait until the connection is established, the timeout is `None`.
        // Otherwise, we use a zero duration timeout, i.e. we return immediately
        // (but we still go through the scheduler once -- which is fine).
        let timeout = if should_wait {
            None
        } else {
            Some((TimeoutClock::Monotonic, TimeoutAnchor::Absolute, Duration::ZERO))
        };

        this.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Write,
            timeout,
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<Socket>,
                    should_wait: bool,
                    foreign_name: &'static str,
                    action: DynMachineCallback<'tcx, Result<(), ()>>,
                } |this, kind: UnblockKind| {
                    // Remove the blocking I/O interest for unblocking this thread.
                    this.machine.blocking_io.remove_blocked_thread(socket.id(), this.machine.threads.active_thread());

                    if UnblockKind::TimedOut == kind {
                        // We can only time out when `should_wait` is false.
                        // This then means that the socket is not yet connected.
                        assert!(!should_wait);
                        return action.call(this, Err(()))
                    }

                    // The thread woke up because it's ready, indicating a writeable or error event.

                    let mut state = socket.state.borrow_mut();
                    let stream = match &*state {
                        SocketState::Connecting(stream) => stream,
                        SocketState::Connected(_) => {
                            drop(state);
                            // This can happen because we blocked the thread:
                            // maybe another thread "upgraded" the connection in the meantime.
                            return action.call(this, Ok(()))
                        },
                        _ => {
                            drop(state);
                            // We ensured that we only block when we're currently connecting.
                            // Since this thread just got rescheduled, it could be that another
                            // thread realized that the connection failed and we're thus in
                            // an "invalid state".
                            return action.call(this, Err(()))
                        }
                    };

                    // Manually check whether there were any errors since calling `connect`.
                    if let Ok(Some(err)) = stream.take_error() {
                        // There was an error during connecting and thus we
                        // return ENOTCONN. It's the program's responsibility
                        // to read SO_ERROR itself.

                        // Store the error such that we can return it when
                        // `getsockopt(SOL_SOCKET, SO_ERROR, ...)` is called on the socket.
                        socket.error.replace(Some(err));

                        // Go back to initial state since the only way of getting into the
                        // `Connecting` state is from the `Initial` state and at this point
                        // we know that the connection won't be established anymore.
                        *state = SocketState::Initial;
                        drop(state);
                        return action.call(this, Err(()))
                    }

                    // There was no error during connecting. Mio advises also reading the peer address
                    // to ensure that socket is actually connected and that it wasn't a spurious wake-up:
                    // <https://docs.rs/mio/latest/mio/net/struct.TcpStream.html#notes>
                    //
                    // Attempting to read the peer address would introduce an edge-case where the
                    // write end of the socket could already be shutdown before it received a
                    // writable event. When we then call [`TcpStream::peer_addr`] we receive an
                    // error. This would need extra state for storing whether the write end was
                    // manually closed using `shutdown`.
                    // Also, tokio doesn't read the peer address and everything seems to be fine,
                    // so we don't do that either:
                    // <https://github.com/tokio-rs/mio/issues/1942#issuecomment-4162607761>
                    // In other words, we are assuming that there will be no spurious
                    // wakeups while establishing the connection.

                    // The connection is established.

                    // Temporarily use dummy state to take ownership of the stream.
                    let SocketState::Connecting(stream) = std::mem::replace(&mut*state, SocketState::Initial) else {
                        // At the start of the function we ensured that we're currently connecting.
                        unreachable!()
                    };
                    *state = SocketState::Connected(stream);
                    drop(state);
                    action.call(this, Ok(()))
                }
            ),
        )
    }
}

impl VisitProvenance for FileDescriptionRef<Socket> {
    // A socket doesn't contain any references to machine memory
    // and thus we don't need to propagate the visit.
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {}
}

impl SourceFileDescription for Socket {
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()> {
        let mut state = self.state.borrow_mut();
        match &mut *state {
            SocketState::Listening(listener) => f(listener),
            SocketState::Connecting(stream) | SocketState::Connected(stream) => f(stream),
            // We never try adding a socket which is not backed by a real socket to the poll registry.
            _ => unreachable!(),
        }
    }

    fn get_readiness_mut(&self) -> RefMut<'_, BlockingIoSourceReadiness> {
        self.io_readiness.borrow_mut()
    }
}
