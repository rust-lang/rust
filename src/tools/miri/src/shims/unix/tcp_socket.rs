use std::cell::{Cell, RefCell, RefMut};
use std::io;
use std::io::Read;
use std::net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use mio::event::Source;
use mio::net::{TcpListener, TcpStream};
use rustc_const_eval::interpret::{InterpResult, interp_ok};
use rustc_middle::throw_unsup_format;
use rustc_target::spec::Os;

use crate::shims::files::{EvalContextExt as _, FdNum, FileDescription, FileDescriptionRef};
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::socket::{SocketFamily, UnixSocketFileDescription};
use crate::*;

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
    /// The SO_ERROR socket option has been set after calling
    /// the `connect` syscall, indicating that the connection
    /// attempt failed. By the POSIX specification, a socket is
    /// is an unspecified state after a failed connection attempt
    /// and thus nothing (except destroying the socket) should be
    /// supported when a socket is in this state.
    ConnectionFailed(TcpStream),
}

#[derive(Debug)]
pub(super) struct TcpSocket {
    /// Family of the socket, used to ensure socket only binds/connects to address of
    /// same family.
    family: SocketFamily,
    /// Current state of the inner socket.
    state: RefCell<SocketState>,
    /// Whether this fd is non-blocking or not.
    is_non_block: Cell<bool>,
    /// The current blocking I/O readiness of the file description.
    io_readiness: RefCell<Readiness>,
    /// [`Some`] when the socket had an async error which has not yet been fetched via `SO_ERROR`.
    error: RefCell<Option<io::Error>>,
    /// Read timeout of the socket. [`None`] means that reads can block indefinitely.
    /// The timeout is applied to the monotonic clock (the Unix specification doesn't
    /// specify which clock to use, but the monotonic clock is more common for
    /// relative timeouts).
    /// This is ignored when the socket is non-blocking.
    read_timeout: Cell<Option<Duration>>,
    /// Write timeout of the socket. [`None`] means that writes can block indefinitely.
    /// The timeout is applied to the monotonic clock (the Unix specification doesn't
    /// specify which clock to use, but the monotonic clock is more common
    /// for relative timeouts).
    /// This is ignored when the socket is non-blocking.
    write_timeout: Cell<Option<Duration>>,
    /// State for being watched by epoll.
    watched: ReadinessWatched,
}

impl TcpSocket {
    pub fn new(family: SocketFamily, is_non_block: bool) -> Self {
        TcpSocket {
            family,
            state: RefCell::new(SocketState::Initial),
            is_non_block: Cell::new(is_non_block),
            io_readiness: RefCell::new(Readiness::EMPTY),
            error: RefCell::new(None),
            read_timeout: Cell::new(None),
            write_timeout: Cell::new(None),
            watched: ReadinessWatched::default(),
        }
    }
}

impl FileDescription for TcpSocket {
    fn name(&self) -> &'static str {
        "socket"
    }

    fn read<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        self.recv(
            communicate_allowed,
            ptr,
            len,
            /* is_peek */ false,
            /* is_non_block */ false,
            ecx,
            finish,
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
        self.send(communicate_allowed, ptr, len, /* is_non_block */ false, ecx, finish)
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

    fn readiness_watched(&self) -> Option<&ReadinessWatched> {
        Some(&self.watched)
    }

    fn readiness(&self) -> Readiness {
        self.io_readiness.borrow().clone()
    }
}

impl UnixFileDescription for TcpSocket {
    fn ioctl<'tcx>(
        &self,
        op: Scalar,
        arg: Option<&OpTy<'tcx>>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, i32> {
        assert!(ecx.machine.communicate(), "cannot have `TcpSocket` with isolation enabled!");

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

    fn as_socket<'tcx>(
        self: FileDescriptionRef<Self>,
        _ecx: &MiriInterpCx<'tcx>,
    ) -> Option<FileDescriptionRef<dyn UnixSocketFileDescription>> {
        Some(self)
    }
}

impl UnixSocketFileDescription for TcpSocket {
    fn bind<'tcx>(
        self: FileDescriptionRef<TcpSocket>,
        communicate_allowed: bool,
        address: SocketAddr,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");
        ecx.ensure_not_failed(&self, "bind")?;

        let mut state = self.state.borrow_mut();

        match *state {
            SocketState::Initial => {
                let address_family = match &address {
                    SocketAddr::V4(_) => SocketFamily::IPv4,
                    SocketAddr::V6(_) => SocketFamily::IPv6,
                };

                if self.family != address_family {
                    // Attempted to bind an address from a family that doesn't match
                    // the family of the socket.
                    let err = if matches!(ecx.tcx.sess.target.os, Os::Linux | Os::Android) {
                        // Linux man page states that `EINVAL` is used when there is an address family mismatch.
                        // See <https://man7.org/linux/man-pages/man2/bind.2.html>
                        LibcError("EINVAL")
                    } else {
                        // POSIX man page states that `EAFNOSUPPORT` should be used when there is an address
                        // family mismatch.
                        // See <https://man7.org/linux/man-pages/man3/bind.3p.html>
                        LibcError("EAFNOSUPPORT")
                    };
                    return interp_ok(Err(err));
                }

                *state = SocketState::Bound(address);
            }
            SocketState::Connecting(_) | SocketState::Connected(_) =>
                throw_unsup_format!(
                    "bind: tcp socket is already connected and binding a
                   connected socket is unsupported"
                ),
            SocketState::Bound(_) | SocketState::Listening(_) =>
                throw_unsup_format!(
                    "bind: tcp socket is already bound and binding a socket \
                   multiple times is unsupported"
                ),
            SocketState::ConnectionFailed(_) => unreachable!(),
        }

        interp_ok(Ok(()))
    }

    fn listen<'tcx>(
        self: FileDescriptionRef<TcpSocket>,
        communicate_allowed: bool,
        // Since the backlog value is just a performance hint we can ignore it.
        _backlog: i32,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");
        ecx.ensure_not_failed(&self, "listen")?;

        let mut state = self.state.borrow_mut();

        match *state {
            SocketState::Bound(socket_addr) =>
                match TcpListener::bind(socket_addr) {
                    Ok(listener) => {
                        *state = SocketState::Listening(listener);
                        drop(state);
                        // Register the socket to the blocking I/O manager because
                        // we now have an associated host socket.
                        ecx.machine.blocking_io.register(self);
                    }
                    Err(e) => return interp_ok(Err(IoError::HostError(e))),
                },
            SocketState::Initial => {
                throw_unsup_format!(
                    "listen: listening on a tcp socket which isn't bound is unsupported"
                )
            }
            SocketState::Listening(_) => {
                throw_unsup_format!(
                    "listen: listening on a tcp socket multiple times is unsupported"
                )
            }
            SocketState::Connecting(_) | SocketState::Connected(_) => {
                throw_unsup_format!("listen: listening on a connected tcp socket is unsupported")
            }
            SocketState::ConnectionFailed(_) => unreachable!(),
        }

        interp_ok(Ok(()))
    }

    fn accept<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        is_client_sock_non_block: bool,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<(FdNum, SocketAddr), IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");

        if !matches!(*self.state.borrow(), SocketState::Listening(_)) {
            throw_unsup_format!(
                "accept: accepting incoming connections is only allowed when tcp socket is listening"
            )
        };

        if self.is_non_block.get() {
            // We have a non-blocking socket and thus don't want to block until
            // we can accept an incoming connection.
            let result = ecx.try_non_block_accept(&self, is_client_sock_non_block)?;
            finish.call(ecx, result)
        } else {
            // The socket is in blocking mode and thus the accept call should block
            // until an incoming connection is ready.

            if self.read_timeout.get().is_some() {
                // Some Unixes like Linux also apply the SO_RCVTIMEO socket option
                // to `accept` calls:
                // <https://github.com/torvalds/linux/blob/HEAD/net/ipv4/inet_connection_sock.c#L668-L675>
                // This is currently not supported by Miri.
                throw_unsup_format!(
                    "accept: blocking tcp accept is not supported when SO_RCVTIMEO is non-zero"
                )
            }

            ecx.block_for_accept(self, is_client_sock_non_block, finish)
        }
    }

    fn connect<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        address: SocketAddr,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<(), IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");
        ecx.ensure_not_failed(&self, "connect")?;

        match &*self.state.borrow() {
            SocketState::Initial => { /* fall-through to below */ }
            // The socket is already in a connecting state.
            SocketState::Connecting(_) => return finish.call(ecx, Err(LibcError("EALREADY"))),
            // We don't return EISCONN for already connected sockets, for which we're
            // sure that the connection is established, since TCP sockets are usually
            // allowed to be connected multiple times.
            _ =>
                throw_unsup_format!(
                    "connect: connecting is only supported for tcp sockets which are neither \
                   bound, listening nor already connected"
                ),
        }

        // This begins establishing the connection, but does not block until the stream is fully connected.
        // We deal with that below.
        match TcpStream::connect(address) {
            Ok(stream) => {
                *self.state.borrow_mut() = SocketState::Connecting(stream);
                // Register the socket to the blocking I/O manager because
                // we now have an associated host socket.
                ecx.machine.blocking_io.register(self.clone());
            }
            Err(e) => return finish.call(ecx, Err(IoError::HostError(e))),
        };

        if self.is_non_block.get() {
            // We have a non-blocking socket and thus don't want to block until
            // the connection is established.

            // Since the [`TcpStream::connect`] function of mio hides the EINPROGRESS
            // we just always return EINPROGRESS and check whether the connection succeeded
            // once we want to use the connected socket.
            finish.call(ecx, Err(LibcError("EINPROGRESS")))
        } else {
            // The socket is in blocking mode and thus the connect call should block
            // until the connection with the server is established.

            if self.write_timeout.get().is_some() {
                // Some Unixes like Linux also apply the SO_SNDTIMEO socket option
                // to `connect` calls:
                // <https://github.com/torvalds/linux/blob/HEAD/net/ipv4/af_inet.c#L701-L710>
                // This is currently not supported by Miri.
                throw_unsup_format!(
                    "connect: blocking connect is not supported when SO_SNDTIMEO is non-zero"
                )
            }

            let socket = self;
            ecx.ensure_connected(
                socket.clone(),
                /* deadline */ None,
                "connect",
                callback!(
                    @capture<'tcx> {
                        socket: FileDescriptionRef<TcpSocket>,
                        finish: DynMachineCallback<'tcx, Result<(), IoError>>,
                    } |this, result: Result<(), ()>| {
                        if result.is_err() {
                            // An error occurred whilst connecting. We know
                            // that it has been consumed by `ensure_connected`
                            // and is now stored in `socket.error`.
                            let err = socket.error.take().unwrap();
                            finish.call(this, Err(IoError::HostError(err)))
                        } else {
                            finish.call(this, Ok(()))
                        }
                    }
                ),
            )
        }
    }

    fn send<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        is_non_block: bool,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");

        let is_non_block = is_non_block || self.is_non_block.get();
        let deadline = ecx.action_deadline(is_non_block, self.write_timeout.get());

        let socket = self;
        ecx.ensure_connected(
            socket.clone(),
            deadline.clone(),
            "send",
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<TcpSocket>,
                    deadline: Option<Deadline>,
                    ptr: Pointer,
                    len: usize,
                    is_non_block: bool,
                    finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                } |this, result: Result<(), ()>| {
                    if result.is_err() {
                        return finish.call(this, Err(LibcError("ENOTCONN")))
                    }

                    if is_non_block {
                        // We have a non-blocking operation or a non-blocking socket and
                        // thus don't want to block until we can send.
                        let result = this.try_non_block_send(&socket, ptr, len)?;
                        finish.call(this, result)
                    } else {
                        // The socket is in blocking mode and thus the send call should block
                        // until we can send some bytes into the socket or the timeout exceeded.
                        this.block_for_send(socket, deadline, ptr, len, finish)
                    }
                }
            ),
        )
    }

    fn recv<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        is_peek: bool,
        is_non_block: bool,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");

        let is_non_block = is_non_block || self.is_non_block.get();
        let deadline = ecx.action_deadline(is_non_block, self.read_timeout.get());

        let socket = self;
        ecx.ensure_connected(
            socket.clone(),
            deadline.clone(),
            "recv",
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<TcpSocket>,
                    deadline: Option<Deadline>,
                    ptr: Pointer,
                    len: usize,
                    is_peek: bool,
                    is_non_block: bool,
                    finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                } |this, result: Result<(), ()>| {
                    if result.is_err() {
                        return finish.call(this, Err(LibcError("ENOTCONN")))
                    }

                    if is_non_block {
                        // We have a non-blocking operation or a non-blocking socket and
                        // thus don't want to block until we can receive.
                        let result = this.try_non_block_recv(&socket, ptr, len, is_peek)?;
                        finish.call(this, result)
                    } else {
                        // The socket is in blocking mode and thus the receive call should block
                        // until we can receive some bytes from the socket or the timeout exceeded.
                        this.block_for_recv(socket, deadline, ptr, len, is_peek, finish)
                    }
                }
            ),
        )
    }

    fn setsockopt<'tcx>(
        self: FileDescriptionRef<Self>,
        level: i32,
        option: i32,
        value_ptr: Pointer,
        value_len: u64,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        if level == ecx.eval_libc_i32("SOL_SOCKET") {
            let opt_so_rcvtimeo = ecx.eval_libc_i32("SO_RCVTIMEO");
            let opt_so_sndtimeo = ecx.eval_libc_i32("SO_SNDTIMEO");
            let opt_so_reuseaddr = ecx.eval_libc_i32("SO_REUSEADDR");

            if matches!(ecx.tcx.sess.target.os, Os::MacOs | Os::FreeBsd | Os::NetBsd) {
                // SO_NOSIGPIPE only exists on MacOS, FreeBSD, and NetBSD.
                let opt_so_nosigpipe = ecx.eval_libc_i32("SO_NOSIGPIPE");

                if option == opt_so_nosigpipe {
                    if value_len != 4 {
                        // Option value should be C-int which is usually 4 bytes.
                        return interp_ok(Err(LibcError("EINVAL")));
                    }
                    let option_value = ecx.ptr_to_mplace(value_ptr, ecx.machine.layouts.i32);
                    let _val = ecx.read_scalar(&option_value)?.to_i32()?;
                    // We entirely ignore this value since we do not support signals anyway.

                    return interp_ok(Ok(()));
                }
            }

            if option == opt_so_rcvtimeo || option == opt_so_sndtimeo {
                let timeval_layout = ecx.libc_ty_layout("timeval");
                let option_value = ecx.ptr_to_mplace(value_ptr, timeval_layout);

                let timeout = match ecx.read_timeval(&option_value)? {
                    None => return interp_ok(Err(LibcError("EINVAL"))),
                    Some(Duration::ZERO) => None,
                    Some(duration) => Some(duration),
                };

                if option == opt_so_rcvtimeo {
                    self.read_timeout.set(timeout);
                } else {
                    self.write_timeout.set(timeout);
                }

                return interp_ok(Ok(()));
            }

            if option == opt_so_reuseaddr {
                if value_len != 4 {
                    // Option value should be C-int which is usually 4 bytes.
                    return interp_ok(Err(LibcError("EINVAL")));
                }
                let option_value = ecx.ptr_to_mplace(value_ptr, ecx.machine.layouts.i32);
                let _val = ecx.read_scalar(&option_value)?.to_i32()?;
                // We entirely ignore this: std always sets REUSEADDR for us, and in the end it's more of a
                // hint to bypass some arbitrary timeout anyway.
                return interp_ok(Ok(()));
            } else {
                throw_unsup_format!(
                    "setsockopt: option {option:#x} is unsupported for level SOL_SOCKET",
                );
            }
        } else if level == ecx.eval_libc_i32("IPPROTO_IP") {
            let opt_ip_ttl = ecx.eval_libc_i32("IP_TTL");

            if option == opt_ip_ttl {
                if value_len != 4 {
                    // Option value should be C-uint which is usually 4 bytes.
                    return interp_ok(Err(LibcError("EINVAL")));
                }
                let option_value = ecx.ptr_to_mplace(value_ptr, ecx.machine.layouts.u32);
                let ttl = ecx.read_scalar(&option_value)?.to_u32()?;

                let result = match &*self.state.borrow() {
                    SocketState::Initial | SocketState::Bound(_) =>
                        throw_unsup_format!(
                            "setsockopt: setting option IP_TTL on level IPPROTO_IP is only supported \
                           on connected and listening tcp sockets"
                        ),
                    SocketState::Listening(listener) => listener.set_ttl(ttl),
                    SocketState::Connecting(stream) | SocketState::Connected(stream) =>
                        stream.set_ttl(ttl),
                    SocketState::ConnectionFailed(_) => unreachable!(),
                };

                return match result {
                    Ok(_) => interp_ok(Ok(())),
                    Err(e) => interp_ok(Err(IoError::HostError(e))),
                };
            } else {
                throw_unsup_format!(
                    "setsockopt: option {option:#x} is unsupported for level IPPROTO_IP",
                );
            }
        } else if level == ecx.eval_libc_i32("IPPROTO_TCP") {
            let opt_tcp_nodelay = ecx.eval_libc_i32("TCP_NODELAY");

            if option == opt_tcp_nodelay {
                if value_len != 4 {
                    // Option value should be C-int which is usually 4 bytes.
                    return interp_ok(Err(LibcError("EINVAL")));
                }
                let option_value = ecx.ptr_to_mplace(value_ptr, ecx.machine.layouts.i32);
                let nodelay = ecx.read_scalar(&option_value)?.to_i32()? != 0;

                let result = match &*self.state.borrow() {
                    SocketState::Initial | SocketState::Bound(_) | SocketState::Listening(_) =>
                        throw_unsup_format!(
                            "setsockopt: setting option TCP_NODELAY on level IPPROTO_TCP is only supported \
                           on connected tcp sockets"
                        ),
                    SocketState::Connecting(stream) | SocketState::Connected(stream) =>
                        stream.set_nodelay(nodelay),
                    SocketState::ConnectionFailed(_) => unreachable!(),
                };

                return match result {
                    Ok(_) => interp_ok(Ok(())),
                    Err(e) => interp_ok(Err(IoError::HostError(e))),
                };
            } else {
                throw_unsup_format!(
                    "setsockopt: option {option:#x} is unsupported for level IPPROTO_TCP"
                );
            }
        }

        throw_unsup_format!(
            "setsockopt: level {level:#x} is unsupported, only SOL_SOCKET, IPPROTO_IP \
           and IPPROTO_TCP are allowed"
        );
    }

    fn getsockopt<'tcx>(
        self: FileDescriptionRef<Self>,
        level: i32,
        option: i32,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<MPlaceTy<'tcx>, IoError>> {
        if level == ecx.eval_libc_i32("SOL_SOCKET") {
            let opt_so_error = ecx.eval_libc_i32("SO_ERROR");
            let opt_so_rcvtimeo = ecx.eval_libc_i32("SO_RCVTIMEO");
            let opt_so_sndtimeo = ecx.eval_libc_i32("SO_SNDTIMEO");

            if option == opt_so_error {
                // Reading SO_ERROR should always return the latest async error. Because our stored
                // `socket.error` could be outdated, we attempt to update it here.
                ecx.update_last_error(&self);

                let return_value = match self.error.take() {
                    Some(err) => ecx.io_error_to_errnum(err)?.to_i32()?,
                    // If there is no error, we return 0 as the option value.
                    None => 0,
                };

                // Clear our own stored error -- it was either `take`n above or it is outdated.
                self.error.replace(None);

                // We know there is no longer an async error and thus we need to update the
                // I/O and fd readiness of the socket.
                self.io_readiness.borrow_mut().error = false;
                ecx.update_fd_readiness(self, /* force_edge */ false)?;

                // Allocate new buffer on the stack with the `i32` layout.
                let value_buffer = ecx.allocate(ecx.machine.layouts.i32, MemoryKind::Stack)?;
                ecx.write_int(return_value, &value_buffer)?;
                interp_ok(Ok(value_buffer))
            } else if option == opt_so_rcvtimeo || option == opt_so_sndtimeo {
                let timeout = if option == opt_so_rcvtimeo {
                    self.read_timeout.get()
                } else {
                    self.write_timeout.get()
                }
                .unwrap_or_default();

                let secs = timeout.as_secs();
                let usecs = timeout.subsec_micros();

                let timeval_layout = ecx.libc_ty_layout("timeval");
                // Allocate new buffer on the stack with the `timeval` layout.
                let timeval_buffer = ecx.allocate(timeval_layout, MemoryKind::Stack)?;

                let sec_field = ecx.project_field_named(&timeval_buffer, "tv_sec")?;
                ecx.write_int(secs, &sec_field)?;

                let usec_field = ecx.project_field_named(&timeval_buffer, "tv_usec")?;
                ecx.write_int(usecs, &usec_field)?;

                interp_ok(Ok(timeval_buffer))
            } else {
                throw_unsup_format!(
                    "getsockopt: option {option:#x} is unsupported for level SOL_SOCKET",
                );
            }
        } else if level == ecx.eval_libc_i32("IPPROTO_IP") {
            let opt_ip_ttl = ecx.eval_libc_i32("IP_TTL");

            if option == opt_ip_ttl {
                let ttl = match &*self.state.borrow() {
                    SocketState::Initial | SocketState::Bound(_) =>
                        throw_unsup_format!(
                            "getsockopt: reading option IP_TTL on level IPPROTO_IP is only supported \
                            on connected and listening tcp sockets"
                        ),
                    SocketState::Listening(listener) => listener.ttl(),
                    SocketState::Connecting(stream) | SocketState::Connected(stream) =>
                        stream.ttl(),
                    SocketState::ConnectionFailed(_) => unreachable!(),
                };

                let ttl = match ttl {
                    Ok(ttl) => ttl,
                    Err(e) => return interp_ok(Err(IoError::HostError(e))),
                };

                // Allocate new buffer on the stack with the `u32` layout.
                let value_buffer = ecx.allocate(ecx.machine.layouts.u32, MemoryKind::Stack)?;
                ecx.write_int(ttl, &value_buffer)?;
                interp_ok(Ok(value_buffer))
            } else {
                throw_unsup_format!(
                    "getsockopt: option {option:#x} is unsupported for level IPPROTO_IP",
                );
            }
        } else if level == ecx.eval_libc_i32("IPPROTO_TCP") {
            let opt_tcp_nodelay = ecx.eval_libc_i32("TCP_NODELAY");

            if option == opt_tcp_nodelay {
                let nodelay = match &*self.state.borrow() {
                    SocketState::Initial | SocketState::Bound(_) | SocketState::Listening(_) =>
                        throw_unsup_format!(
                            "getsockopt: reading option TCP_NODELAY on level IPPROTO_TCP is only supported \
                            on connected tcp sockets"
                        ),
                    SocketState::Connecting(stream) | SocketState::Connected(stream) =>
                        stream.nodelay(),
                    SocketState::ConnectionFailed(_) => unreachable!(),
                };

                let nodelay = match nodelay {
                    Ok(nodelay) => nodelay,
                    Err(e) => return interp_ok(Err(IoError::HostError(e))),
                };

                // Allocate new buffer on the stack with the `i32` layout.
                let value_buffer = ecx.allocate(ecx.machine.layouts.i32, MemoryKind::Stack)?;
                ecx.write_int(i32::from(nodelay), &value_buffer)?;
                interp_ok(Ok(value_buffer))
            } else {
                throw_unsup_format!(
                    "getsockopt: option {option:#x} is unsupported for level IPPROTO_TCP"
                );
            }
        } else {
            throw_unsup_format!(
                "getsockopt: level {level:#x} is unsupported, only SOL_SOCKET, IPPROTO_IP \
               and IPPROTO_TCP are allowed"
            )
        }
    }

    fn getsockname<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<SocketAddr, IoError>> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");
        ecx.ensure_not_failed(&self, "getsockname")?;

        let state = self.state.borrow();

        let address = match &*state {
            SocketState::Bound(address) => {
                if address.port() == 0 {
                    // The socket is bound to a zero-port which means it gets assigned a random
                    // port. Since we don't yet have an underlying socket, we don't know what this
                    // random port will be and thus this is unsupported.
                    throw_unsup_format!(
                        "getsockname: when the port is 0, getting the tcp socket address before \
                        calling `listen` or `connect` is unsupported"
                    )
                }

                *address
            }
            SocketState::Listening(listener) =>
                match listener.local_addr() {
                    Ok(address) => address,
                    Err(e) => return interp_ok(Err(IoError::HostError(e))),
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
                        ecx.emit_diagnostic(NonHaltingDiagnostic::ConnectingSocketGetsockname);
                    }
                }
                match stream.local_addr() {
                    Ok(address) => address,
                    Err(e) => return interp_ok(Err(IoError::HostError(e))),
                }
            }
            // For non-bound sockets the POSIX manual says the returned address is unspecified.
            // Often this is 0.0.0.0:0 and thus we set it to this value.
            SocketState::Initial => SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)),
            SocketState::ConnectionFailed(_) => unreachable!(),
        };

        interp_ok(Ok(address))
    }

    fn getpeername<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<SocketAddr, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");

        let socket = self;
        // It's only safe to call [`TcpStream::peer_addr`] after the socket is connected since
        // UNIX targets should return ENOTCONN when the connection is not yet established.
        ecx.ensure_connected(
            socket.clone(),
            // Check whether the socket is connected without blocking.
            Some(ecx.machine.monotonic_clock.now().into()),
            "getpeername",
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<TcpSocket>,
                    finish: DynMachineCallback<'tcx, Result<SocketAddr, IoError>>,
                } |this, result: Result<(), ()>| {
                    if result.is_err() {
                        return finish.call(this, Err(LibcError("ENOTCONN")))
                    };

                    let SocketState::Connected(stream) = &*socket.state.borrow() else {
                        unreachable!()
                    };

                    let result = stream.peer_addr().map_err(IoError::HostError);
                    finish.call(this, result)
                }
            ),
        )
    }

    fn shutdown<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        how: Shutdown,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        assert!(communicate_allowed, "cannot have `TcpSocket` with isolation enabled!");
        ecx.ensure_not_failed(&self, "shutdown")?;

        let state = self.state.borrow();

        let (SocketState::Connecting(stream) | SocketState::Connected(stream)) = &*state else {
            return interp_ok(Err(LibcError("ENOTCONN")));
        };

        if let Err(e) = stream.shutdown(how) {
            return interp_ok(Err(IoError::HostError(e)));
        };

        drop(state);

        // Because we map cross platform mio readiness to our readiness struct and
        // the different platforms don't treat `shutdown` the same way, we set
        // the readiness after a `shutdown` manually to achieve a more consistent
        // readiness. Otherwise we do not generate enough readiness events
        // on partial shutdowns on Windows hosts.
        let mut readiness = self.io_readiness.borrow_mut();
        // Closing the read end of a socket causes an (E)POLLRDHUP event.
        readiness.read_closed |= matches!(how, Shutdown::Read | Shutdown::Both);
        // Only shutting down the write end doesn't cause an (E)POLLHUP event
        // and thus we won't set the `write_closed` readiness for it here.
        readiness.write_closed |= matches!(how, Shutdown::Both);
        // The Linux kernel also sets EPOLLIN when the read end of a socket is closed:
        // <https://github.com/torvalds/linux/blob/HEAD/net/ipv4/tcp.c#L584-L588>
        readiness.readable |= matches!(how, Shutdown::Read | Shutdown::Both);

        drop(readiness);

        // Update the readiness for the socket.
        ecx.update_fd_readiness(self, /* force_edge */ false)?;

        interp_ok(Ok(()))
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Get the deadline for an action (e.g. reading or writing).
    /// When `is_non_block` is [`true`], the returned deadline is "now", i.e.,
    /// we wake up immediately if the action cannot be completed.
    /// If `action_timeout` is `Some(duration)`, the returned deadline is in the
    /// future be the specified `duration`. Otherwise, no deadline ([`None`]) is
    /// returned, indicating that the action can block indefinitely.
    fn action_deadline(
        &self,
        is_non_block: bool,
        action_timeout: Option<Duration>,
    ) -> Option<Deadline> {
        let this = self.eval_context_ref();

        if is_non_block {
            // Non-blocking sockets always have a zero timeout.
            Some(this.machine.monotonic_clock.now().into())
        } else {
            action_timeout
                .map(|duration| this.machine.monotonic_clock.now().add_lossy(duration).into())
        }
    }

    /// Block the thread until there's an incoming connection or an error occurred.
    /// After a successful accept, `finish` is called with a tuple containing the
    /// file descriptor of the peer socket and it's address.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    ///
    /// **Note**: This function is only safe to call when having previously ensured
    /// that the socket is in [`SocketState::Listening`].
    fn block_for_accept(
        &mut self,
        socket: FileDescriptionRef<TcpSocket>,
        is_client_sock_nonblock: bool,
        finish: DynMachineCallback<'tcx, Result<(FdNum, SocketAddr), IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Since the callback holds a strong reference to the socket, the file description
        // won't be closed as long as some thread is blocked on it. While this reflects
        // what Linux does, for other Unix systems this might differ from the native behavior.
        this.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Read,
            /* deadline */ None,
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<TcpSocket>,
                is_client_sock_nonblock: bool,
                finish: DynMachineCallback<'tcx, Result<(FdNum, SocketAddr), IoError>>,
            } |this, kind: UnblockKind| {
                // Remove the blocking I/O interest for unblocking this thread.
                this.machine.blocking_io.remove_blocked_thread(socket.id(), this.machine.threads.active_thread());

                match kind {
                    UnblockKind::Ready => { /* fall-through to below */ },
                    // When the read timeout is exceeded EAGAIN/EWOULDBLOCK is returned.
                    UnblockKind::TimedOut => return finish.call(this, Err(LibcError("EWOULDBLOCK")))
                }

                match this.try_non_block_accept(&socket, is_client_sock_nonblock)? {
                    Ok((sockfd, addr)) => finish.call(this, Ok((sockfd, addr))),
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        this.block_for_accept(socket, is_client_sock_nonblock, finish)
                    }
                    Err(e) => finish.call(this, Err(e)),
                }
            }),
        )
    }

    /// Attempt to accept an incoming connection on the listening socket in a
    /// non-blocking manner. After a successful accept, a tuple containing the
    /// file descriptor of the peer socket and it's address is returned.
    ///
    /// **Note**: This function is only safe to call when having previously ensured
    /// that the socket is in [`SocketState::Listening`].
    fn try_non_block_accept(
        &mut self,
        socket: &FileDescriptionRef<TcpSocket>,
        is_client_sock_nonblock: bool,
    ) -> InterpResult<'tcx, Result<(FdNum, SocketAddr), IoError>> {
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
                this.update_fd_readiness(socket.clone(), /* force_edge */ false)?;

                return interp_ok(Err(IoError::HostError(e)));
            }
            Err(e) => return interp_ok(Err(IoError::HostError(e))),
        };

        let family = match addr {
            SocketAddr::V4(_) => SocketFamily::IPv4,
            SocketAddr::V6(_) => SocketFamily::IPv6,
        };

        let fd = this.machine.fds.new_ref(TcpSocket {
            family,
            state: RefCell::new(SocketState::Connected(stream)),
            is_non_block: Cell::new(is_client_sock_nonblock),
            io_readiness: RefCell::new(Readiness::EMPTY),
            error: RefCell::new(None),
            read_timeout: Cell::new(None),
            write_timeout: Cell::new(None),
            watched: ReadinessWatched::default(),
        });
        // Register the socket to the blocking I/O manager because
        // there is an associated host socket.
        this.machine.blocking_io.register(fd.clone());
        let sockfd = this.machine.fds.insert(fd);
        interp_ok(Ok((sockfd, addr)))
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
        socket: FileDescriptionRef<TcpSocket>,
        deadline: Option<Deadline>,
        buffer_ptr: Pointer,
        length: usize,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Since the callback holds a strong reference to the socket, the file description
        // won't be closed as long as some thread is blocked on it. While this reflects
        // what Linux does, for other Unix systems this might differ from the native behavior.
        this.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Write,
            deadline.clone(),
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<TcpSocket>,
                deadline: Option<Deadline>,
                buffer_ptr: Pointer,
                length: usize,
                finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
            } |this, kind: UnblockKind| {
                // Remove the blocking I/O interest for unblocking this thread.
                this.machine.blocking_io.remove_blocked_thread(socket.id(), this.machine.threads.active_thread());

                match kind {
                    UnblockKind::Ready => { /* fall-through to below */ },
                    // When the write timeout is exceeded EAGAIN/EWOULDBLOCK is returned.
                    UnblockKind::TimedOut => return finish.call(this, Err(LibcError("EWOULDBLOCK")))
                }

                match this.try_non_block_send(&socket, buffer_ptr, length)? {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        this.block_for_send(socket, deadline, buffer_ptr, length, finish)
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
        socket: &FileDescriptionRef<TcpSocket>,
        buffer_ptr: Pointer,
        length: usize,
    ) -> InterpResult<'tcx, Result<usize, IoError>> {
        let this = self.eval_context_mut();

        let mut state = socket.state.borrow_mut();
        let SocketState::Connected(stream) = &mut *state else {
            panic!("try_non_block_send must only be called when the socket is connected")
        };

        // This is a *non-blocking* write.
        let result = this.write_to_host(stream, length, buffer_ptr)?;

        drop(state);

        // A write should never succeed when the `write_closed` readiness is set for this socket.
        if result.is_ok() {
            assert!(!socket.io_readiness.borrow().write_closed, "successful write after close");
        }

        match result {
            Err(IoError::HostError(e))
                if matches!(e.kind(), io::ErrorKind::NotConnected | io::ErrorKind::WouldBlock) =>
            {
                // We know that the source is not writable so we need to update its readiness.
                socket.io_readiness.borrow_mut().writable = false;
                this.update_fd_readiness(socket.clone(), /* force_edge */ false)?;

                // On Windows hosts, `send` can return WSAENOTCONN where EAGAIN or EWOULDBLOCK
                // would be returned on UNIX-like systems. We thus remap this error to an EWOULDBLOCK.
                interp_ok(Err(IoError::HostError(io::ErrorKind::WouldBlock.into())))
            }
            Ok(bytes_written) if bytes_written < length => {
                // We had a short write. On Unix hosts using the `epoll` and `kqueue` backends, a
                // short write means that the write buffer is full. We update the readiness
                // accordingly, which means that next time we see "writable" we will report an
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
                    this.update_fd_readiness(socket.clone(), /* force_edge */ false)?;
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
                    this.update_fd_readiness(socket.clone(), /* force_edge */ true)?;
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
        socket: FileDescriptionRef<TcpSocket>,
        deadline: Option<Deadline>,
        buffer_ptr: Pointer,
        length: usize,
        should_peek: bool,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Since the callback holds a strong reference to the socket, the file description
        // won't be closed as long as some thread is blocked on it. While this reflects
        // what Linux does, for other Unix systems this might differ from the native behavior.
        this.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Read,
            deadline.clone(),
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<TcpSocket>,
                deadline: Option<Deadline>,
                buffer_ptr: Pointer,
                length: usize,
                should_peek: bool,
                finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
            } |this, kind: UnblockKind| {
                // Remove the blocking I/O interest for unblocking this thread.
                this.machine.blocking_io.remove_blocked_thread(socket.id(), this.machine.threads.active_thread());

                match kind {
                    UnblockKind::Ready => { /* fall-through to below */ },
                    // When the read timeout is exceeded EAGAIN/EWOULDBLOCK is returned.
                    UnblockKind::TimedOut => return finish.call(this, Err(LibcError("EWOULDBLOCK")))
                }

                match this.try_non_block_recv(&socket, buffer_ptr, length, should_peek)? {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        this.block_for_recv(socket, deadline, buffer_ptr, length, should_peek, finish)
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
        socket: &FileDescriptionRef<TcpSocket>,
        buffer_ptr: Pointer,
        length: usize,
        should_peek: bool,
    ) -> InterpResult<'tcx, Result<usize, IoError>> {
        let this = self.eval_context_mut();

        let mut state = socket.state.borrow_mut();
        let SocketState::Connected(stream) = &mut *state else {
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

        drop(state);

        match result {
            Err(IoError::HostError(e))
                if matches!(e.kind(), io::ErrorKind::NotConnected | io::ErrorKind::WouldBlock) =>
            {
                // We know that the source is not readable so we need to update its readiness.
                socket.io_readiness.borrow_mut().readable = false;
                this.update_fd_readiness(socket.clone(), /* force_edge */ false)?;

                // On Windows hosts, `recv` can return WSAENOTCONN where EAGAIN or EWOULDBLOCK
                // would be returned on UNIX-like systems. We thus remap this error to an EWOULDBLOCK.
                interp_ok(Err(IoError::HostError(io::ErrorKind::WouldBlock.into())))
            }
            Ok(bytes_read)
                if !should_peek
                    && bytes_read < length
                    && bytes_read > 0
                    && !socket.io_readiness.borrow().read_closed =>
            {
                // We had a short read (and were not peeking). (Note that reading 0 bytes is guaranteed
                // to indicate EOF, and can never happen spuriously, so we have to exclude that case.
                // We also don't want to clear the readable readiness for sockets whose read end has
                // already been closed as those never block a read, i.e., they are always read-ready.)
                // On Unix hosts using the `epoll` and `kqueue` backends, a short read means that the
                // read buffer is empty. We update the readiness accordingly, which means that next time
                // we see "readable" we will report an edge. Some applications (e.g. tokio) rely on
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
                    this.update_fd_readiness(socket.clone(), /* force_edge */ false)?;
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
                    this.update_fd_readiness(socket.clone(), /* force_edge */ true)?;
                }
                interp_ok(result)
            }
            result => interp_ok(result),
        }
    }

    // Execute the provided callback function when the socket is either in
    // [`SocketState::Connected`] or an error occurred.
    /// If the socket is currently neither in the [`SocketState::Connecting`] nor
    /// the [`SocketState::Connecting`] state, [`Err`] is returned.
    /// When the callback function is called with [`Ok`], then we're guaranteed
    /// that the socket is in the [`SocketState::Connected`] state.
    ///
    /// This method internally calls `ensure_not_failed` and thus an unsupported
    /// error is thrown should `socket` be in [`SocketState::ConnectionFailed`].
    ///
    /// This function can optionally also block until either an error occurred or
    /// the socket reached the [`SocketState::Connected`] state.
    fn ensure_connected(
        &mut self,
        socket: FileDescriptionRef<TcpSocket>,
        deadline: Option<Deadline>,
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
                this.ensure_not_failed(&socket, foreign_name)?;
                return action.call(this, Err(()));
            }
        };

        drop(state);

        // We're currently connecting. Since the underlying mio socket is non-blocking,
        // the only way to determine whether we are done connecting is by polling.

        this.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Write,
            deadline,
            callback!(
                @capture<'tcx> {
                    socket: FileDescriptionRef<TcpSocket>,
                    foreign_name: &'static str,
                    action: DynMachineCallback<'tcx, Result<(), ()>>,
                } |this, kind: UnblockKind| {
                    // Remove the blocking I/O interest for unblocking this thread.
                    this.machine.blocking_io.remove_blocked_thread(socket.id(), this.machine.threads.active_thread());

                    if UnblockKind::TimedOut == kind {
                        // This then means that the socket is not yet connected.
                        return action.call(this, Err(()))
                    }

                    // The thread woke up because it's ready, indicating a writeable or error event.

                    let state = socket.state.borrow();
                    match &*state {
                        SocketState::Connecting(_) => { /* fall-through to below */ },
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
                            this.ensure_not_failed(&socket, foreign_name)?;
                            return action.call(this, Err(()))
                        }
                    };

                    drop(state);

                    // Set `socket.error` if `socket` currently has an error.
                    this.update_last_error(&socket);

                    if socket.error.borrow().is_some() {
                        // There was an error during connecting.
                        // It's the program's responsibility to read SO_ERROR itself.
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
                    let mut state = socket.state.borrow_mut();
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

    /// Ensure that `socket` is not in the [`SocketState::ConnectionFailed`] state.
    /// If `socket` is currently in [`SocketState::ConnectionFailed`], an unsupported
    /// error is thrown.
    fn ensure_not_failed(
        &self,
        socket: &FileDescriptionRef<TcpSocket>,
        foreign_name: &'static str,
    ) -> InterpResult<'tcx> {
        if let SocketState::ConnectionFailed(_) = &*socket.state.borrow() {
            throw_unsup_format!(
                "{foreign_name}: sockets are in an unspecified state after a failed `connect`; \
                any operation on such a socket is thus unsupported"
            );
        } else {
            interp_ok(())
        }
    }

    /// Check whether the underlying host socket of `socket` contains an error.
    /// If there is an error, we store it in `socket.error`.
    ///
    /// Should `socket` be in the [`SocketState::Connecting`] state whilst there is
    /// an error on the host socket, we transition into the [`SocketState::ConnectionFailed`]
    /// state because we know that `socket` can no longer successfully establish a
    /// connection.
    fn update_last_error(&self, socket: &FileDescriptionRef<TcpSocket>) {
        let mut state = socket.state.borrow_mut();

        let new_error = match &*state {
            SocketState::Listening(listener) =>
                listener.take_error().expect("Reading SO_ERROR should not fail"),
            SocketState::Connecting(stream) | SocketState::Connected(stream) =>
                stream.take_error().expect("Reading SO_ERROR should not fail"),
            SocketState::Initial | SocketState::Bound(_) | SocketState::ConnectionFailed(_) => None,
        };

        let Some(new_error) = new_error else { return };

        // Store the error such that we can return it when
        // `getsockopt(SOL_SOCKET, SO_ERROR, ...)` is called on the socket.
        socket.error.replace(Some(new_error));

        if matches!(&*state, SocketState::Connecting(_)) {
            // After reading an error on a connecting socket, we know that
            // the connection won't be established anymore. By the POSIX
            // specification, the socket is now in an unspecified state.
            // We thus change the socket state to `ConnectionFailed`.

            // Temporarily use dummy state to take ownership of the stream.
            let SocketState::Connecting(stream) =
                std::mem::replace(&mut *state, SocketState::Initial)
            else {
                unreachable!()
            };
            *state = SocketState::ConnectionFailed(stream);
        }
    }
}

impl SourceFileDescription for TcpSocket {
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()> {
        let mut state = self.state.borrow_mut();
        match &mut *state {
            SocketState::Listening(listener) => f(listener),
            SocketState::Connecting(stream)
            | SocketState::Connected(stream)
            | SocketState::ConnectionFailed(stream) => f(stream),
            // We never try adding a socket which is not backed by a real socket to the poll registry.
            _ => unreachable!(),
        }
    }

    fn get_readiness_mut(&self) -> RefMut<'_, Readiness> {
        self.io_readiness.borrow_mut()
    }
}
