use std::cell::{Cell, RefCell};
use std::io::Read;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use std::time::Duration;
use std::{io, iter};

use mio::Interest;
use mio::event::Source;
use mio::net::{TcpListener, TcpStream};
use rustc_abi::Size;
use rustc_const_eval::interpret::{InterpResult, interp_ok};
use rustc_middle::throw_unsup_format;
use rustc_target::spec::Os;

use crate::concurrency::blocking_io::InterestReceiver;
use crate::shims::files::{EvalContextExt as _, FdId, FileDescription, FileDescriptionRef};
use crate::shims::unix::UnixFileDescription;
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
                        this.block_for_recv(socket, ptr, len, /* should_peek */ false, finish);
                        interp_ok(())
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
                        this.block_for_send(socket, ptr, len, finish);
                        interp_ok(())
                    }
                }
            ),
        )
    }

    fn short_fd_operations(&self) -> bool {
        // Linux de-facto guarantees (or at least, applications like tokio assume [1, 2]) that
        // when a read/write on a streaming socket comes back short, the kernel buffer is
        // empty/full. SO we can't do short reads/writes here.
        //
        // [1]: https://github.com/tokio-rs/tokio/blob/6c03e03898d71eca976ee1ad8481cf112ae722ba/tokio/src/io/poll_evented.rs#L182
        // [2]: https://github.com/tokio-rs/tokio/blob/6c03e03898d71eca976ee1ad8481cf112ae722ba/tokio/src/io/poll_evented.rs#L240
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
            );
            interp_ok(())
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
        // don't return an error after receiving an [`Interest::WRITEABLE`]
        // event on the stream.
        match TcpStream::connect(address) {
            Ok(stream) => *socket.state.borrow_mut() = SocketState::Connecting(stream),
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
                        );
                        interp_ok(())
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
                        );
                        interp_ok(())
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

                    match this.write_socket_address(
                        &address,
                        address_ptr,
                        address_len_ptr,
                        "getpeername",
                    )? {
                        Ok(_) => this.write_scalar(Scalar::from_i32(0), &dest),
                        Err(e) => this.set_last_error_and_return(e, &dest),
                    }
                }
            ),
        )
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
                        this.block_for_accept(socket, address_ptr, address_len_ptr, is_client_sock_nonblock, dest);
                        interp_ok(())
                    }
                    Err(e) => this.set_last_error_and_return(e, &dest),
                }
            }),
        );
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
            if let Err(e) =
                this.write_socket_address(&addr, address_ptr, address_len_ptr, "accept4")?
            {
                return interp_ok(Err(e));
            };
        }

        let fd = this.machine.fds.new_ref(Socket {
            family,
            state: RefCell::new(SocketState::Connected(stream)),
            is_non_block: Cell::new(is_client_sock_nonblock),
        });
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

                match this.try_non_block_send(&socket, buffer_ptr, length)? {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        this.block_for_send(socket, buffer_ptr, length, finish);
                        interp_ok(())
                    },
                    result => finish.call(this, result)
                }
            }),
        );
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
        // FIXME: When the host does a short write, we should emit an epoll edge -- at least for targets for which tokio assumes no short writes:
        // <https://github.com/tokio-rs/tokio/blob/6c03e03898d71eca976ee1ad8481cf112ae722ba/tokio/src/io/poll_evented.rs#L240>
        match result {
            Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::NotConnected => {
                // On Windows hosts, `send` can return WSAENOTCONN where EAGAIN or EWOULDBLOCK
                // would be returned on UNIX-like systems. We thus remap this error to an EWOULDBLOCK.
                interp_ok(Err(IoError::HostError(io::ErrorKind::WouldBlock.into())))
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

                match this.try_non_block_recv(&socket, buffer_ptr, length, should_peek)? {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        this.block_for_recv(socket, buffer_ptr, length, should_peek, finish);
                        interp_ok(())
                    },
                    result => finish.call(this, result)
                }
            }),
        );
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
        // FIXME: When the host does a short read, we should emit an epoll edge -- at least for targets for which tokio assumes no short reads:
        // <https://github.com/tokio-rs/tokio/blob/6c03e03898d71eca976ee1ad8481cf112ae722ba/tokio/src/io/poll_evented.rs#L182>
        match result {
            Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::NotConnected => {
                // On Windows hosts, `recv` can return WSAENOTCONN where EAGAIN or EWOULDBLOCK
                // would be returned on UNIX-like systems. We thus remap this error to an EWOULDBLOCK.
                interp_ok(Err(IoError::HostError(io::ErrorKind::WouldBlock.into())))
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
            Interest::WRITABLE,
            timeout,
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<Socket>,
                    should_wait: bool,
                    foreign_name: &'static str,
                    action: DynMachineCallback<'tcx, Result<(), ()>>,
                } |this, kind: UnblockKind| {
                    if UnblockKind::TimedOut == kind {
                        // We can only time out when `should_wait` is false.
                        // This then means that the socket is not yet connected.
                        assert!(!should_wait);
                        this.machine.blocking_io.deregister(socket.id(), InterestReceiver::UnblockThread(this.active_thread()));
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
                    if let Ok(Some(_)) = stream.take_error() {
                        // There was an error during connecting and thus we
                        // return ENOTCONN. It's the program's responsibility
                        // to read SO_ERROR itself.
                        //
                        // Go back to initial state since the only way of getting into the
                        // `Connecting` state is from the `Initial` state and at this point
                        // we know that the connection won't be established anymore.
                        //
                        // FIXME: We're currently just dropping the error information. Eventually
                        // we'll have to store it so that it can be recovered by the user.
                        *state = SocketState::Initial;
                        drop(state);
                        return action.call(this, Err(()))
                    }

                    // There was no error during connecting. We still need to ensure that
                    // the wakeup wasn't spurious. We do this by attempting to read the
                    // peer address of the socket (following the advice given by mio):
                    // <https://docs.rs/mio/latest/mio/net/struct.TcpStream.html#notes>

                    match stream.peer_addr() {
                        Ok(_) => { /* fall-through to below */},
                        Err(e) if matches!(e.kind(), io::ErrorKind::NotConnected | io::ErrorKind::InProgress) => {
                            // We received a spurious wakeup from the OS. This should be considered an OS bug:
                            // <https://github.com/tokio-rs/mio/issues/1942#issuecomment-4169378308>
                            panic!("{foreign_name}: received writable event from OS but socket is not yet connected")
                        },
                        Err(_) => {
                            // For all other errors the socket is connected. Since we're not interested in the
                            // peer address and only want to know whether the socket is connected, we can ignore
                            // the error and continue.
                        }
                    }

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
        );

        interp_ok(())
    }
}

impl VisitProvenance for FileDescriptionRef<Socket> {
    // A socket doesn't contain any references to machine memory
    // and thus we don't need to propagate the visit.
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {}
}

impl WithSource for Socket {
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()> {
        let mut state = self.state.borrow_mut();
        match &mut *state {
            SocketState::Listening(listener) => f(listener),
            SocketState::Connecting(stream) | SocketState::Connected(stream) => f(stream),
            // We never try adding a socket which is not backed by a real socket to the poll registry.
            _ => unreachable!(),
        }
    }
}
