use std::cell::{Cell, RefCell};
use std::io::Read;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use std::{io, iter};

use mio::Interest;
use mio::event::Source;
use mio::net::{TcpListener, TcpStream};
use rand::Rng;
use rustc_abi::Size;
use rustc_const_eval::interpret::{InterpResult, interp_ok};
use rustc_middle::throw_unsup_format;
use rustc_target::spec::Os;

use crate::shims::files::{EvalContextExt as _, FdId, FileDescription, FileDescriptionRef};
use crate::{OpTy, Scalar, *};

#[derive(Debug, PartialEq)]
enum SocketFamily {
    // IPv4 internet protocols
    IPv4,
    // IPv6 internet protocols
    IPv6,
}

enum SocketIoError {
    /// The socket is not yet ready. Either EINPROGRESS or ENOTCONNECTED occurred.
    NotReady,
    /// Any other kind of I/O error.
    Other(io::Error),
}

impl From<io::Error> for SocketIoError {
    fn from(value: io::Error) -> Self {
        match value.kind() {
            io::ErrorKind::InProgress | io::ErrorKind::NotConnected => Self::NotReady,
            _ => Self::Other(value),
        }
    }
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

impl SocketState {
    /// If the socket is currently in [`SocketState::Connecting`], try to ensure
    /// that the connection is established by first checking that [`TcpStream::take_error`]
    /// doesn't return an error and then by checking that [`TcpStream::peer_addr`]
    /// returns the address of the connected peer.
    ///
    /// If the connection is established or the socket is in any other state,
    /// [`Ok`] is returned.
    ///
    /// **Important**: On Windows hosts this function can only be used to ensure a socket is connected
    /// _after_ a [`Interest::WRITABLE`] event was received.
    pub fn try_set_connected(&mut self) -> Result<(), SocketIoError> {
        // Further explanation of the limitation on Windows hosts:
        // Windows treats sockets which are connecting as connected until either the connection timeout hits
        // or an error occurs. Thus, the [`TcpStream::peer_addr`] method returns [`Ok`] with the provided peer
        // address even when the connection might not yet be established.

        let SocketState::Connecting(stream) = self else { return Ok(()) };

        if let Ok(Some(e)) = stream.take_error() {
            // There was an error whilst connecting.
            let e = SocketIoError::from(e);
            // We won't get EINPROGRESS or ENOTCONNECTED here
            // so we need to reset the state.
            assert!(matches!(e, SocketIoError::Other(_)));
            // Go back to initial state as the only way of getting into the
            // `Connecting` state is from the `Initial` state.
            *self = SocketState::Initial;
            return Err(e);
        }

        if let Err(e) = stream.peer_addr() {
            let e = SocketIoError::from(e);
            if let SocketIoError::Other(_) = &e {
                // All other errors are fatal for a socket and thus the state needs to be reset.
                *self = SocketState::Initial;
            }
            return Err(e);
        };

        // We just read the peer address without an error so we can be
        // sure that the connection is established.

        // Temporarily use dummy state to take ownership of the stream.
        let SocketState::Connecting(stream) = std::mem::replace(self, SocketState::Initial) else {
            // At the start of the function we ensured that we're currently connecting.
            unreachable!()
        };
        *self = SocketState::Connected(stream);
        Ok(())
    }
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
}

impl FileDescription for Socket {
    fn name(&self) -> &'static str {
        "socket"
    }

    fn destroy<'tcx>(
        self,
        _self_id: FdId,
        communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, std::io::Result<()>> {
        assert!(communicate_allowed, "cannot have `Socket` with isolation enabled!");

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

        if !matches!(&*self.state.borrow(), SocketState::Connected(_)) {
            // We can only receive from connected sockets. For all other
            // states we return a not connected error.
            return finish.call(ecx, Err(LibcError("ENOTCONN")));
        }

        // Since `read` is the same as `recv` with no flags, we just treat
        // the `read` as a `recv` here.
        ecx.block_for_recv(self, ptr, len, /* should_peek */ false, finish);

        interp_ok(())
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

        if !matches!(&*self.state.borrow(), SocketState::Connected(_)) {
            // We can only send with connected sockets. For all other
            // states we return a not connected error.
            return finish.call(ecx, Err(LibcError("ENOTCONN")));
        }

        // Since `write` is the same as `send` with no flags, we just treat
        // the `write` as a `send` here.
        ecx.block_for_send(self, ptr, len, finish);

        interp_ok(())
    }

    fn short_fd_operations(&self) -> bool {
        // Short accesses on TCP sockets are realistic and expected to happen.
        true
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
        mut _flag: i32,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        throw_unsup_format!("fcntl: socket flags aren't supported")
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
        if protocol != 0 {
            throw_unsup_format!(
                "socket: socket protocol {protocol} is unsupported, \
                only 0 is allowed"
            );
        }

        let fds = &mut this.machine.fds;
        let fd = fds.new_ref(Socket {
            family,
            state: RefCell::new(SocketState::Initial),
            is_non_block: Cell::new(is_sock_nonblock),
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
        let address = match this.socket_address(address, address_len, "bind")? {
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
                    Ok(listener) => *state = SocketState::Listening(listener),
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
            throw_unsup_format!("accept4: non-blocking accept is unsupported")
        }

        // The socket is in blocking mode and thus the accept call should block
        // until an incoming connection is ready.
        this.block_for_accept(
            address_ptr,
            address_len_ptr,
            is_client_sock_nonblock,
            socket,
            dest.clone(),
        );
        interp_ok(())
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
        let address = match this.socket_address(address, address_len, "connect")? {
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

        // Mio returns a potentially unconnected stream.
        // We can be ensured that the connection is established when
        // [`TcpStream::take_err`] and [`TcpStream::peer_addr`] both
        // don't return errors.
        // For non-blocking sockets we need to check that for every
        // [`Interest::WRITEABLE`] event on the stream.
        match TcpStream::connect(address) {
            Ok(stream) => *socket.state.borrow_mut() = SocketState::Connecting(stream),
            Err(e) => return this.set_last_error_and_return(e, dest),
        };

        if socket.is_non_block.get() {
            throw_unsup_format!("connect: non-blocking connect is unsupported");
        }

        // The socket is in blocking mode and thus the connect call should block
        // until the connection with the server is established.
        this.block_for_connect(socket, dest.clone());
        interp_ok(())
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

        if !matches!(&*socket.state.borrow(), SocketState::Connected(_)) {
            // We can only send with connected sockets. For all other
            // states we return a not connected error.
            return this.set_last_error_and_return(LibcError("ENOTCONN"), dest);
        }

        // Non-deterministically decide to further reduce the length, simulating a partial send.
        // We avoid reducing the write size to 0: the docs seem to be entirely fine with that,
        // but the standard library is not (https://github.com/rust-lang/rust/issues/145959).
        let length = if this.machine.short_fd_operations
            && length >= 2
            && this.machine.rng.get_mut().random()
        {
            length / 2
        } else {
            length
        };

        // Interpret the flag. Every flag we recognize is "subtracted" from `flags`, so
        // if there is anything left at the end, that's an unsupported flag.
        if matches!(
            this.tcx.sess.target.os,
            Os::Linux | Os::Android | Os::FreeBsd | Os::Solaris | Os::Illumos
        ) {
            // MSG_NOSIGNAL only exists on Linux, Android, FreeBSD,
            // Solaris, and Illumos targets.
            let msg_nosignal = this.eval_libc_i32("MSG_NOSIGNAL");
            if flags & msg_nosignal == msg_nosignal {
                // This is only needed to ensure that no EPIPE signal is sent when
                // trying to send into a stream which is no longer connected.
                // Since we don't support signals, we can ignore this.
                flags &= !msg_nosignal;
            }
        }

        if flags != 0 {
            throw_unsup_format!(
                "send: flag {flags:#x} is unsupported, only MSG_NOSIGNAL is allowed",
            );
        }

        let dest = dest.clone();

        this.block_for_send(
            socket,
            buffer_ptr,
            length,
            callback!(@capture<'tcx> {
                dest: MPlaceTy<'tcx>
            } |this, result: Result<usize, IoError>| {
                match result {
                    Ok(read_size) => {
                        let read_size: u64 = read_size.try_into().unwrap();
                        let ssize_layout = this.libc_ty_layout("ssize_t");
                        this.write_scalar(Scalar::from_int(read_size, ssize_layout.size), &dest)
                    }
                    Err(e) => this.set_last_error_and_return(e, &dest)
                }
            }),
        );

        interp_ok(())
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

        if !matches!(&*socket.state.borrow(), SocketState::Connected(_)) {
            // We can only receive from connected sockets. For all other
            // states we return a not connected error.
            return this.set_last_error_and_return(LibcError("ENOTCONN"), dest);
        }

        // Non-deterministically decide to further reduce the length, simulating a partial receive.
        // We don't simulate partial receives for lengths < 2 because the man page states that a
        // return value of zero can only be returned in some special cases:
        // "When a stream socket peer has performed an orderly shutdown, the return value will be 0
        // (the traditional "end-of-file" return). [...] The value 0 may also be returned if the
        // requested number of bytes to receive from a stream socket was 0."
        let length = if this.machine.short_fd_operations
            && length >= 2
            && this.machine.rng.get_mut().random()
        {
            length / 2 // since `length` is at least 2, the result is still at least 1
        } else {
            length
        };

        let mut should_peek = false;

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

        if flags != 0 {
            throw_unsup_format!(
                "recv: flag {flags:#x} is unsupported, only MSG_PEEK \
                and MSG_CMSG_CLOEXEC are allowed",
            );
        }

        let dest = dest.clone();

        this.block_for_recv(
            socket,
            buffer_ptr,
            length,
            should_peek,
            callback!(@capture<'tcx> {
                dest: MPlaceTy<'tcx>
            } |this, result: Result<usize, IoError>| {
                match result {
                    Ok(read_size) => {
                        let read_size: u64 = read_size.try_into().unwrap();
                        let ssize_layout = this.libc_ty_layout("ssize_t");
                        this.write_scalar(Scalar::from_int(read_size, ssize_layout.size), &dest)
                    }
                    Err(e) => this.set_last_error_and_return(e, &dest)
                }
            }),
        );

        interp_ok(())
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
            // For non-bound sockets the POSIX manual says the returned address is unspecified.
            // Often this is 0.0.0.0:0 and thus we set it to this value.
            _ => SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)),
        };

        match this.write_socket_address(&address, address_ptr, address_len_ptr, "getsockname")? {
            Ok(_) => interp_ok(Scalar::from_i32(0)),
            Err(e) => this.set_last_error_and_return_i32(e),
        }
    }

    fn getpeername(
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

        let SocketState::Connected(stream) = &*state else {
            // We can only read the peer address of connected sockets.
            return this.set_last_error_and_return_i32(LibcError("ENOTCONN"));
        };

        let address = match stream.peer_addr() {
            Ok(address) => address,
            Err(e) => return this.set_last_error_and_return_i32(e),
        };

        match this.write_socket_address(&address, address_ptr, address_len_ptr, "getpeername")? {
            Ok(_) => interp_ok(Scalar::from_i32(0)),
            Err(e) => this.set_last_error_and_return_i32(e),
        }
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Attempt to turn an address and length operand into a standard library socket address.
    ///
    /// Returns an IO error should the address length not match the address family length.
    fn socket_address(
        &self,
        address: &OpTy<'tcx>,
        address_len: &OpTy<'tcx>,
        foreign_name: &'static str,
    ) -> InterpResult<'tcx, Result<SocketAddr, IoError>> {
        let this = self.eval_context_ref();

        let socklen_layout = this.libc_ty_layout("socklen_t");
        // We only support address lengths which can be stored in a u64 since the
        // size of a layout is also stored in a u64.
        let address_len: u64 =
            this.read_scalar(address_len)?.to_int(socklen_layout.size)?.try_into().unwrap();

        // Initially, treat address as generic sockaddr just to extract the family field.
        let sockaddr_layout = this.libc_ty_layout("sockaddr");
        if address_len < sockaddr_layout.size.bytes() {
            // Address length should be at least as big as the generic sockaddr
            return interp_ok(Err(LibcError("EINVAL")));
        }
        let address = this.deref_pointer_as(address, sockaddr_layout)?;

        let family_field = this.project_field_named(&address, "sa_family")?;
        let family_layout = this.libc_ty_layout("sa_family_t");
        let family = this.read_scalar(&family_field)?.to_int(family_layout.size)?;

        // Depending on the family, decide whether it's IPv4 or IPv6 and use specialized layout
        // to extract address and port.
        let socket_addr = if family == this.eval_libc_i32("AF_INET").into() {
            let sockaddr_in_layout = this.libc_ty_layout("sockaddr_in");
            if address_len != sockaddr_in_layout.size.bytes() {
                // Address length should be exactly the length of an IPv4 address.
                return interp_ok(Err(LibcError("EINVAL")));
            }
            let address = address.transmute(sockaddr_in_layout, this)?;

            let port_field = this.project_field_named(&address, "sin_port")?;
            // Read bytes and treat them as big endian since port is stored in network byte order.
            let port_bytes: [u8; 2] = this
                .read_bytes_ptr_strip_provenance(port_field.ptr(), Size::from_bytes(2))?
                .try_into()
                .unwrap();
            let port = u16::from_be_bytes(port_bytes);

            let addr_field = this.project_field_named(&address, "sin_addr")?;
            let s_addr_field = this.project_field_named(&addr_field, "s_addr")?;
            // Read bytes and treat them as big endian since address is stored in network byte order.
            let addr_bytes: [u8; 4] = this
                .read_bytes_ptr_strip_provenance(s_addr_field.ptr(), Size::from_bytes(4))?
                .try_into()
                .unwrap();
            let addr_bits = u32::from_be_bytes(addr_bytes);

            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::from_bits(addr_bits), port))
        } else if family == this.eval_libc_i32("AF_INET6").into() {
            let sockaddr_in6_layout = this.libc_ty_layout("sockaddr_in6");
            if address_len != sockaddr_in6_layout.size.bytes() {
                // Address length should be exactly the length of an IPv6 address.
                return interp_ok(Err(LibcError("EINVAL")));
            }
            // We cannot transmute since the `sockaddr_in6` layout is bigger than the `sockaddr` layout.
            let address = address.offset(Size::ZERO, sockaddr_in6_layout, this)?;

            let port_field = this.project_field_named(&address, "sin6_port")?;
            // Read bytes and treat them as big endian since port is stored in network byte order.
            let port_bytes: [u8; 2] = this
                .read_bytes_ptr_strip_provenance(port_field.ptr(), Size::from_bytes(2))?
                .try_into()
                .unwrap();
            let port = u16::from_be_bytes(port_bytes);

            let addr_field = this.project_field_named(&address, "sin6_addr")?;
            let s_addr_field = this
                .project_field_named(&addr_field, "s6_addr")?
                .transmute(this.machine.layouts.u128, this)?;
            // Read bytes and treat them as big endian since address is stored in network byte order.
            let addr_bytes: [u8; 16] = this
                .read_bytes_ptr_strip_provenance(s_addr_field.ptr(), Size::from_bytes(16))?
                .try_into()
                .unwrap();
            let addr_bits = u128::from_be_bytes(addr_bytes);

            let flowinfo_field = this.project_field_named(&address, "sin6_flowinfo")?;
            // flowinfo doesn't get the big endian treatment as this field is stored in native byte order
            // and not in network byte order.
            let flowinfo = this.read_scalar(&flowinfo_field)?.to_u32()?;

            let scope_id_field = this.project_field_named(&address, "sin6_scope_id")?;
            // scope_id doesn't get the big endian treatment as this field is stored in native byte order
            // and not in network byte order.
            let scope_id = this.read_scalar(&scope_id_field)?.to_u32()?;

            SocketAddr::V6(SocketAddrV6::new(
                Ipv6Addr::from_bits(addr_bits),
                port,
                flowinfo,
                scope_id,
            ))
        } else {
            // Socket of other types shouldn't be created in a first place and
            // thus also no address family of another type should be supported.
            throw_unsup_format!(
                "{foreign_name}: address family {family:#x} is unsupported, \
                only AF_INET and AF_INET6 are allowed"
            );
        };

        interp_ok(Ok(socket_addr))
    }

    /// Attempt to write a standard library socket address into a pointer.
    ///
    /// The `address_len_ptr` parameter serves both as input and output parameter.
    /// On input, it points to the size of the buffer `address_ptr` points to, and
    /// on output it points to the non-truncated size of the written address in the
    /// buffer pointed to by `address_ptr`.
    ///
    /// If the address buffer doesn't fit the whole address, the address is truncated to not
    /// overflow the buffer.
    fn write_socket_address(
        &mut self,
        address: &SocketAddr,
        address_ptr: Pointer,
        address_len_ptr: Pointer,
        foreign_name: &'static str,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        let this = self.eval_context_mut();

        if address_ptr == Pointer::null() || address_len_ptr == Pointer::null() {
            // The POSIX man page doesn't account for the cases where the `address_ptr` or
            // `address_len_ptr` could be null pointers. Thus, this behavior is undefined!
            throw_ub_format!(
                "{foreign_name}: writing a socket address but the address or the length pointer is a null pointer"
            )
        }

        let socklen_layout = this.libc_ty_layout("socklen_t");
        let address_buffer_len_place = this.ptr_to_mplace(address_len_ptr, socklen_layout);
        // We only support buffer lengths which can be stored in a u64 since the
        // size of a layout in bytes is also stored in a u64.
        let address_buffer_len: u64 = this
            .read_scalar(&address_buffer_len_place)?
            .to_int(socklen_layout.size)?
            .try_into()
            .unwrap();

        let (address_buffer, address_layout) = match address {
            SocketAddr::V4(address) => {
                // IPv4 address bytes; already stored in network byte order.
                let address_bytes = address.ip().octets();
                // Port needs to be manually turned into network byte order.
                let port = address.port().to_be();

                let sockaddr_in_layout = this.libc_ty_layout("sockaddr_in");
                // Allocate new buffer on the stack with the `sockaddr_in` layout.
                // We need a temporary buffer as `address_ptr` might not point to a large enough
                // buffer, in which case we have to truncate.
                let address_buffer = this.allocate(sockaddr_in_layout, MemoryKind::Stack)?;
                // Zero the whole buffer as some libc targets have additional fields which we fill
                // with zero bytes (just like the standard library does it).
                this.write_bytes_ptr(
                    address_buffer.ptr(),
                    iter::repeat_n(0, address_buffer.layout.size.bytes_usize()),
                )?;

                let sin_family_field = this.project_field_named(&address_buffer, "sin_family")?;
                // We cannot simply write the `AF_INET` scalar into the `sin_family_field` because on most
                // systems the field has a layout of 16-bit whilst the scalar has a size of 32-bit.
                // Since the `AF_INET` constant is chosen such that it can safely be converted into
                // a 16-bit integer, we use the following logic to get a scalar of the right size.
                let af_inet = this.eval_libc("AF_INET");
                let address_family =
                    Scalar::from_int(af_inet.to_int(af_inet.size())?, sin_family_field.layout.size);
                this.write_scalar(address_family, &sin_family_field)?;

                let sin_port_field = this.project_field_named(&address_buffer, "sin_port")?;
                // Write the port in target native endianness bytes as we already converted it
                // to big endian above.
                this.write_bytes_ptr(sin_port_field.ptr(), port.to_ne_bytes())?;

                let sin_addr_field = this.project_field_named(&address_buffer, "sin_addr")?;
                let s_addr_field = this.project_field_named(&sin_addr_field, "s_addr")?;
                this.write_bytes_ptr(s_addr_field.ptr(), address_bytes)?;

                (address_buffer, sockaddr_in_layout)
            }
            SocketAddr::V6(address) => {
                // IPv6 address bytes; already stored in network byte order.
                let address_bytes = address.ip().octets();
                // Port needs to be manually turned into network byte order.
                let port = address.port().to_be();
                // Flowinfo is stored in native byte order.
                let flowinfo = address.flowinfo();
                // Scope id is stored in native byte order.
                let scope_id = address.scope_id();

                let sockaddr_in6_layout = this.libc_ty_layout("sockaddr_in6");
                // Allocate new buffer on the stack with the `sockaddr_in6` layout.
                // We need a temporary buffer as `address_ptr` might not point to a large enough
                // buffer, in which case we have to truncate.
                let address_buffer = this.allocate(sockaddr_in6_layout, MemoryKind::Stack)?;
                // Zero the whole buffer as some libc targets have additional fields which we fill
                // with zero bytes (just like the standard library does it).
                this.write_bytes_ptr(
                    address_buffer.ptr(),
                    iter::repeat_n(0, address_buffer.layout.size.bytes_usize()),
                )?;

                let sin6_family_field = this.project_field_named(&address_buffer, "sin6_family")?;
                // We cannot simply write the `AF_INET6` scalar into the `sin6_family_field` because on most
                // systems the field has a layout of 16-bit whilst the scalar has a size of 32-bit.
                // Since the `AF_INET6` constant is chosen such that it can safely be converted into
                // a 16-bit integer, we use the following logic to get a scalar of the right size.
                let af_inet6 = this.eval_libc("AF_INET6");
                let address_family = Scalar::from_int(
                    af_inet6.to_int(af_inet6.size())?,
                    sin6_family_field.layout.size,
                );
                this.write_scalar(address_family, &sin6_family_field)?;

                let sin6_port_field = this.project_field_named(&address_buffer, "sin6_port")?;
                // Write the port in target native endianness bytes as we already converted it
                // to big endian above.
                this.write_bytes_ptr(sin6_port_field.ptr(), port.to_ne_bytes())?;

                let sin6_flowinfo_field =
                    this.project_field_named(&address_buffer, "sin6_flowinfo")?;
                this.write_scalar(Scalar::from_u32(flowinfo), &sin6_flowinfo_field)?;

                let sin6_scope_id_field =
                    this.project_field_named(&address_buffer, "sin6_scope_id")?;
                this.write_scalar(Scalar::from_u32(scope_id), &sin6_scope_id_field)?;

                let sin6_addr_field = this.project_field_named(&address_buffer, "sin6_addr")?;
                let s6_addr_field = this.project_field_named(&sin6_addr_field, "s6_addr")?;
                this.write_bytes_ptr(s6_addr_field.ptr(), address_bytes)?;

                (address_buffer, sockaddr_in6_layout)
            }
        };

        // Copy the truncated address into the pointer pointed to by `address_ptr`.
        this.mem_copy(
            address_buffer.ptr(),
            address_ptr,
            // Truncate the address to fit the provided buffer.
            address_layout.size.min(Size::from_bytes(address_buffer_len)),
            // The buffers are guaranteed to not overlap since the `address_buffer`
            // was just newly allocated on the stack.
            true,
        )?;
        // Deallocate the address buffer as it was only needed to construct the address and
        // copy it into the buffer pointed to by `address_ptr`.
        this.deallocate_ptr(address_buffer.ptr(), None, MemoryKind::Stack)?;
        // Size of the non-truncated address.
        let address_len = address_layout.size.bytes();

        this.write_scalar(
            Scalar::from_uint(address_len, socklen_layout.size),
            &address_buffer_len_place,
        )?;

        interp_ok(Ok(()))
    }

    /// Block the thread until there's an incoming connection or an error occurred.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    fn block_for_accept(
        &mut self,
        address_ptr: Pointer,
        address_len_ptr: Pointer,
        is_client_sock_nonblock: bool,
        socket: FileDescriptionRef<Socket>,
        dest: MPlaceTy<'tcx>,
    ) {
        let this = self.eval_context_mut();
        this.block_thread_for_io(
            socket.clone(),
            Interest::READABLE,
            None,
            callback!(@capture<'tcx> {
                address_ptr: Pointer,
                address_len_ptr: Pointer,
                is_client_sock_nonblock: bool,
                socket: FileDescriptionRef<Socket>,
                dest: MPlaceTy<'tcx>,
            } |this, kind: UnblockKind| {
                assert_eq!(kind, UnblockKind::Ready);

                let state = socket.state.borrow();

                let SocketState::Listening(listener) = &*state else {
                    // We checked that the socket is in listening state before blocking
                    // and since there is no outgoing transition from that state this
                    // should be unreachable.
                    unreachable!()
                };

                let (stream, addr) = match listener.accept() {
                    Ok(peer) => peer,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        drop(state);
                        this.block_for_accept(address_ptr, address_len_ptr, is_client_sock_nonblock, socket, dest);
                        return interp_ok(())
                    },
                    Err(e) => return this.set_last_error_and_return(e, &dest),
                };

                let family = match addr {
                    SocketAddr::V4(_) => SocketFamily::IPv4,
                    SocketAddr::V6(_) => SocketFamily::IPv6,
                };

                if address_ptr != Pointer::null() {
                    // We only attempt a write if the address pointer is not a null pointer.
                    // If the address pointer is a null pointer the user isn't interested in the
                    // address and we don't need to write anything.
                    if let Err(e) = this.write_socket_address(&addr, address_ptr, address_len_ptr, "accept4")? {
                      return this.set_last_error_and_return(e, &dest);
                    };
                }

                let fd = this.machine.fds.new_ref(Socket {
                    family,
                    state: RefCell::new(SocketState::Connected(stream)),
                    is_non_block: Cell::new(is_client_sock_nonblock),
                });
                let sockfd = this.machine.fds.insert(fd);
                // We need to create the scalar using the destination size since
                // `syscall(SYS_accept4, ...)` returns a long which doesn't match
                // the int returned from the `accept`/`accept4` syscalls.
                // See <https://man7.org/linux/man-pages/man2/syscall.2.html>.
                this.write_scalar(Scalar::from_int(sockfd, dest.layout.size), &dest)
            }),
        );
    }

    /// Block the thread until the stream is connected or an error occurred.
    fn block_for_connect(&mut self, socket: FileDescriptionRef<Socket>, dest: MPlaceTy<'tcx>) {
        let this = self.eval_context_mut();
        this.block_thread_for_io(
            socket.clone(),
            Interest::WRITABLE,
            None,
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<Socket>,
                dest: MPlaceTy<'tcx>,
            } |this, kind: UnblockKind| {
                assert_eq!(kind, UnblockKind::Ready);

                let mut state = socket.state.borrow_mut();

                // We received a "writable" event so `try_set_connected` is safe to call.
                match state.try_set_connected() {
                    Ok(_) => this.write_scalar(Scalar::from_i32(0), &dest),
                     Err(SocketIoError::NotReady) => {
                        // We need to block the thread again as the connection is still not yet ready.
                        drop(state);
                        this.block_for_connect(socket, dest);
                        return interp_ok(())
                    },
                    Err(SocketIoError::Other(e)) => return this.set_last_error_and_return(e, &dest)
                }
            }),
        );
    }

    /// Block the thread until we can send bytes into the connected socket
    /// or an error occurred.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    fn block_for_send(
        &mut self,
        socket: FileDescriptionRef<Socket>,
        buffer_ptr: Pointer,
        length: usize,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) {
        let this = self.eval_context_mut();
        this.block_thread_for_io(
            socket.clone(),
            Interest::WRITABLE,
            None,
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<Socket>,
                buffer_ptr: Pointer,
                length: usize,
                finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
            } |this, kind: UnblockKind| {
                assert_eq!(kind, UnblockKind::Ready);

                let mut state = socket.state.borrow_mut();
                let SocketState::Connected(stream) = &mut*state else {
                    // We ensured that the socket is connected before blocking.
                    unreachable!()
                };

                // This is a *non-blocking* write.
                let result = this.write_to_host(stream, length, buffer_ptr)?;
                match result {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        drop(state);
                        this.block_for_send(socket, buffer_ptr, length, finish);
                        interp_ok(())
                    },
                    result => finish.call(this, result)
                }
            }),
        );
    }

    /// Block the thread until we can receive bytes from the connected socket
    /// or an error occurred.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    fn block_for_recv(
        &mut self,
        socket: FileDescriptionRef<Socket>,
        buffer_ptr: Pointer,
        length: usize,
        should_peek: bool,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) {
        let this = self.eval_context_mut();
        this.block_thread_for_io(
            socket.clone(),
            Interest::READABLE,
            None,
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<Socket>,
                buffer_ptr: Pointer,
                length: usize,
                should_peek: bool,
                finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
            } |this, kind: UnblockKind| {
                assert_eq!(kind, UnblockKind::Ready);

                let mut state = socket.state.borrow_mut();
                let SocketState::Connected(stream) = &mut*state else {
                    // We ensured that the socket is connected before blocking.
                    unreachable!()
                };

                // This is a *non-blocking* read/peek.
                let result = this.read_from_host(|buf| {
                    if should_peek {
                        stream.peek(buf)
                    } else {
                        stream.read(buf)
                    }
                }, length, buffer_ptr)?;
                match result {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        drop(state);
                        this.block_for_recv(socket, buffer_ptr, length, should_peek, finish);
                        interp_ok(())
                    },
                    result => finish.call(this, result)
                }
            }),
        );
    }
}

impl VisitProvenance for FileDescriptionRef<Socket> {
    // A socket doesn't contain any references to machine memory
    // and thus we don't need to propagate the visit.
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {}
}

impl WithSource for FileDescriptionRef<Socket> {
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()> {
        let mut state = self.state.borrow_mut();
        match &mut *state {
            SocketState::Listening(listener) => f(listener),
            SocketState::Connecting(stream) | SocketState::Connected(stream) => f(stream),
            // We never try adding a socket which is not backed by a real socket to the poll registry.
            _ => unreachable!(),
        }
    }

    fn id(&self) -> FdId {
        self.id()
    }
}
