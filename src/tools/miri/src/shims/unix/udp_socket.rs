use std::cell::{Cell, RefCell, RefMut};
use std::io;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use std::time::Duration;

use mio::event::Source;
use rustc_target::spec::Os;

use crate::shims::files::{EvalContextExt as _, FileDescription, FileDescriptionRef};
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::socket::{
    EvalContextExt, UnixSocketFileDescription, socket_action_deadline,
};
use crate::*;

#[derive(Debug)]
pub(super) struct UdpSocket {
    /// Family of the socket, used to ensure that the socket only binds/connects to addresses
    /// of the same family.
    family: socket2::Domain,
    /// The underlying host socket.
    socket: RefCell<mio::net::UdpSocket>,
    /// Whether this fd is non-blocking or not.
    is_non_block: Cell<bool>,
    /// Whether the socket is implicitly or explicitly bound to an address.
    is_bound: Cell<bool>,
    /// Whether a default destination address has been set for this socket.
    has_default_destination: Cell<bool>,
    /// The current blocking I/O readiness of the file description.
    io_readiness: RefCell<Readiness>,
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

impl UdpSocket {
    pub fn new<'tcx>(
        family: socket2::Domain,
        is_non_block: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> io::Result<FileDescriptionRef<Self>> {
        let socket =
            socket2::Socket::new(family, socket2::Type::DGRAM, Some(socket2::Protocol::UDP))?;
        // The underlying host socket needs to be non-blocking. The actual
        // blocking mode of the socket is stored in `is_non_block`.
        socket.set_nonblocking(true)?;

        let socket = cfg_select! {
            // Turn the `socket2::Socket` into a `mio::UdpSocket`. We cannot construct the
            // mio socket directly, since its only constructor is `UdpSocket::bind`.
            unix => {{
                use std::os::fd::OwnedFd;
                mio::net::UdpSocket::from(OwnedFd::from(socket))
            }},
            windows => {{
                use std::os::windows::io::OwnedSocket;
                mio::net::UdpSocket::from(OwnedSocket::from(socket))
            }},
            _ => unreachable!("unsupported host platform"),
        };

        let udp_socket = UdpSocket {
            family,
            socket: RefCell::new(socket),
            is_non_block: Cell::new(is_non_block),
            is_bound: Cell::new(false),
            has_default_destination: Cell::new(false),
            // We can set the default readiness to the empty readiness
            // since the socket will be registered to the blocking I/O
            // directly after its creation.
            io_readiness: RefCell::new(Readiness::EMPTY),
            read_timeout: Cell::new(None),
            write_timeout: Cell::new(None),
            watched: ReadinessWatched::default(),
        };

        let fd = ecx.machine.fds.new_ref(udp_socket);

        // Add the UDP socket to the blocking I/O manager as it's always
        // backed by a mio UdpSocket.
        ecx.machine.blocking_io.register(fd.clone());

        Ok(fd)
    }
}

impl FileDescription for UdpSocket {
    fn name(&self) -> &'static str {
        "udp socket"
    }

    fn read<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: u64,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<u64, IoError>>,
    ) -> InterpResult<'tcx> {
        self.recv(
            communicate_allowed,
            ptr,
            len,
            /* is_peek */ false,
            /* is_non_block */ false,
            ecx,
            callback!(
                @capture<'tcx> {
                    finish: DynMachineCallback<'tcx, Result<u64, IoError>>,
                } |ecx, result: Result<(u64, Option<SocketAddr>), IoError>| {
                    finish.call(ecx, result.map(|(bytes, _)| bytes))
                }
            ),
        )
    }

    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: u64,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<u64, IoError>>,
    ) -> InterpResult<'tcx> {
        self.send(
            communicate_allowed,
            ptr,
            len,
            /* is_non_block */ false,
            /* dest_addr */ None,
            ecx,
            finish,
        )
    }

    fn as_unix(self: FileDescriptionRef<Self>) -> FileDescriptionRef<dyn UnixFileDescription> {
        self
    }

    fn readiness_watched(&self) -> Option<&ReadinessWatched> {
        Some(&self.watched)
    }

    fn readiness(&self) -> Readiness {
        *self.io_readiness.borrow()
    }
}

impl UnixFileDescription for UdpSocket {
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

    fn as_socket(
        self: FileDescriptionRef<Self>,
    ) -> Option<FileDescriptionRef<dyn UnixSocketFileDescription>> {
        Some(self)
    }
}

impl UnixSocketFileDescription for UdpSocket {
    fn bind<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        address: SocketAddr,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        assert!(communicate_allowed, "cannot have `UdpSocket` with isolation enabled!");

        let address_family = match &address {
            SocketAddr::V4(_) => socket2::Domain::IPV4,
            SocketAddr::V6(_) => socket2::Domain::IPV6,
        };

        if self.family != address_family {
            // Attempted to bind an address from a family that doesn't match
            // the family of the socket.
            let err = if matches!(ecx.tcx.sess.target.os, Os::Linux | Os::Android | Os::MacOs) {
                // Linux man page states that `EINVAL` is used when there is an address family mismatch.
                // See <https://man7.org/linux/man-pages/man2/bind.2.html>
                // macOS also returns `EINVAL` but their man page does not specify this.
                LibcError("EINVAL")
            } else {
                // POSIX man page states that `EAFNOSUPPORT` should be used when there is an address
                // family mismatch.
                // See <https://man7.org/linux/man-pages/man3/bind.3p.html>
                LibcError("EAFNOSUPPORT")
            };
            return interp_ok(Err(err));
        }

        // `bind` is a non-blocking operation for UDP sockets.
        if let Err(e) = self.with_socket_ref(|s| s.bind(&socket2::SockAddr::from(address))).unwrap()
        {
            return interp_ok(Err(IoError::HostError(e)));
        }

        // The socket has been explicitly bound to a local address.
        self.is_bound.set(true);

        interp_ok(Ok(()))
    }

    fn connect<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        address: SocketAddr,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<(), IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `UdpSocket` with isolation enabled!");

        if self.has_default_destination.get() {
            // POSIX is very vague about whether it's allowed to set the default destination address
            // of a non-connection-mode socket multiple times. Some platforms support it while others don't.
            // We thus only allow setting the destination address at most once.
            // See <https://github.com/rust-lang/miri/pull/5507#issuecomment-6779828795>
            throw_unsup_format!("calling `connect` a second time on a UDP socket is unsupported");
        }

        if let Err(e) = self.socket.borrow().connect(address) {
            return finish.call(ecx, Err(IoError::HostError(e)));
        }

        // POSIX specifies that unbound sockets get bound to an unassigned
        // local address after `connect` has been invoked on them.
        // We thus treat the socket as implicitly bound.
        self.is_bound.set(true);

        self.has_default_destination.set(true);

        finish.call(ecx, Ok(()))
    }

    fn send<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: u64,
        is_non_block: bool,
        dest_addr: Option<SocketAddr>,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<u64, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `UdpSocket` with isolation enabled!");

        let is_non_block = is_non_block || self.is_non_block.get();

        if is_non_block {
            // We have a non-blocking operation or a non-blocking socket and
            // thus don't want to block until we can send.
            let result = ecx.try_non_block_send(&self, ptr, len, dest_addr)?;
            finish.call(ecx, result)
        } else {
            let deadline = socket_action_deadline(ecx, is_non_block, self.write_timeout.get());

            // The socket is in blocking mode and thus the send call should block
            // until we can send some bytes into the socket or the timeout exceeded.
            ecx.block_for_send(self, deadline, ptr, len, dest_addr, finish)
        }
    }

    fn recv<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ptr: Pointer,
        len: u64,
        is_peek: bool,
        is_non_block: bool,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<(u64, Option<SocketAddr>), IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `UdpSocket` with isolation enabled!");

        let is_non_block = is_non_block || self.is_non_block.get();

        if is_non_block {
            // We have a non-blocking operation or a non-blocking socket and
            // thus don't want to block until we can receive.
            let result = ecx.try_non_block_recv(&self, ptr, len, is_peek)?;
            finish.call(ecx, result.map(|(bytes, addr)| (bytes, Some(addr))))
        } else {
            let deadline = socket_action_deadline(ecx, is_non_block, self.read_timeout.get());

            // The socket is in blocking mode and thus the receive call should block
            // until we can receive some bytes from the socket or the timeout exceeded.
            ecx.block_for_recv(self, deadline, ptr, len, is_peek, callback!(
                @capture<'tcx> {
                    finish: DynMachineCallback<'tcx, Result<(u64, Option<SocketAddr>), IoError>>,
                } |ecx, result: Result<(u64, SocketAddr), IoError>| {
                    // For UDP sockets we always have access to the address from where the
                    // data originated.
                    finish.call(ecx, result.map(|(bytes, addr)| (bytes, Some(addr))))
                }
            ))
        }
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
            return ecx.set_generic_socket_option(option, value_ptr, value_len, self);
        } else if level == ecx.eval_libc_i32("IPPROTO_IP") {
            let socket = self.socket.borrow();
            return ecx.set_ip_socket_option(
                option,
                value_ptr,
                value_len,
                socket2::SockRef::from(&*socket),
            );
        }

        throw_unsup_format!(
            "setsockopt: level {level:#x} is unsupported, only SOL_SOCKET and IPPROTO_IP \
            are allowed"
        );
    }

    fn getsockopt<'tcx>(
        self: FileDescriptionRef<Self>,
        level: i32,
        option: i32,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<MPlaceTy<'tcx>, IoError>> {
        if level == ecx.eval_libc_i32("SOL_SOCKET") {
            ecx.get_generic_socket_option(option, self)
        } else if level == ecx.eval_libc_i32("IPPROTO_IP") {
            let socket = self.socket.borrow();
            ecx.get_ip_socket_option(option, socket2::SockRef::from(&*socket))
        } else {
            throw_unsup_format!(
                "getsockopt: level {level:#x} is unsupported, only SOL_SOCKET and IPPROTO_IP \
                are allowed"
            )
        }
    }

    fn getsockname<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<SocketAddr, IoError>> {
        assert!(communicate_allowed, "cannot have `UdpSocket` with isolation enabled!");

        if !self.is_bound.get() {
            // Since Windows returns EINVAL when invoking `getsockname` on
            // a socket which hasn't been bound yet, we need to manually
            // return an unspecified address here.

            let address = if self.family == socket2::Domain::IPV4 {
                SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, /* port */ 0))
            } else {
                SocketAddr::V6(SocketAddrV6::new(
                    Ipv6Addr::UNSPECIFIED,
                    /* port */ 0,
                    /* flowinfo */ 0,
                    /* scope_id */ 0,
                ))
            };
            return interp_ok(Ok(address));
        }

        interp_ok(self.socket.borrow().local_addr().map_err(IoError::HostError))
    }

    fn getpeername<'tcx>(
        self: FileDescriptionRef<Self>,
        communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<SocketAddr, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "cannot have `UdpSocket` with isolation enabled!");

        finish.call(ecx, self.socket.borrow().peer_addr().map_err(IoError::HostError))
    }

    fn set_read_timeout<'tcx>(
        self: FileDescriptionRef<Self>,
        timeout: Option<Duration>,
    ) -> InterpResult<'tcx> {
        self.read_timeout.set(timeout);
        interp_ok(())
    }

    fn get_read_timeout<'tcx>(
        self: FileDescriptionRef<Self>,
    ) -> InterpResult<'tcx, Option<Duration>> {
        interp_ok(self.read_timeout.get())
    }

    fn set_write_timeout<'tcx>(
        self: FileDescriptionRef<Self>,
        timeout: Option<Duration>,
    ) -> InterpResult<'tcx> {
        self.write_timeout.set(timeout);
        interp_ok(())
    }

    fn get_write_timeout<'tcx>(
        self: FileDescriptionRef<Self>,
    ) -> InterpResult<'tcx, Option<Duration>> {
        interp_ok(self.write_timeout.get())
    }

    fn with_socket_ref<T>(&self, cb: impl FnOnce(socket2::SockRef<'_>) -> T) -> Result<T, ()>
    where
        Self: Sized,
    {
        let socket = self.socket.borrow();
        Ok(cb(socket2::SockRef::from(&*socket)))
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Block the thread until we can send bytes into the connected socket
    /// or an error occurred.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    fn block_for_send(
        &mut self,
        socket: FileDescriptionRef<UdpSocket>,
        deadline: Option<Deadline>,
        buffer_ptr: Pointer,
        length: u64,
        dest_addr: Option<SocketAddr>,
        finish: DynMachineCallback<'tcx, Result<u64, IoError>>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();
        // Since the callback holds a strong reference to the socket, the file description
        // won't be closed as long as some thread is blocked on it. While this reflects
        // what Linux does, for other Unix systems this might differ from the native behavior.
        ecx.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Write,
            deadline.clone(),
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<UdpSocket>,
                deadline: Option<Deadline>,
                buffer_ptr: Pointer,
                length: u64,
                dest_addr: Option<SocketAddr>,
                finish: DynMachineCallback<'tcx, Result<u64, IoError>>,
            } |ecx, kind: UnblockKind| {
                // Remove the blocking I/O interest for unblocking this thread.
                ecx.machine.blocking_io.remove_blocked_thread(socket.id(), ecx.machine.threads.active_thread());

                match kind {
                    UnblockKind::Ready => { /* fall-through to below */ },
                    // When the write timeout is exceeded EAGAIN/EWOULDBLOCK is returned.
                    UnblockKind::TimedOut => return finish.call(ecx, Err(LibcError("EWOULDBLOCK")))
                }

                match ecx.try_non_block_send(&socket, buffer_ptr, length, dest_addr)? {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        ecx.block_for_send(socket, deadline, buffer_ptr, length, dest_addr, finish)
                    },
                    result => finish.call(ecx, result)
                }
            }),
        )
    }

    /// Attempt to send bytes to another socket. If `dest_addr` is [`Some`], the data
    /// is sent to the address specified in `dest_addr`. Otherwise, the data is sent
    /// to the default destination address.
    fn try_non_block_send(
        &mut self,
        socket: &FileDescriptionRef<UdpSocket>,
        buffer_ptr: Pointer,
        length: u64,
        dest_addr: Option<SocketAddr>,
    ) -> InterpResult<'tcx, Result<u64, IoError>> {
        let ecx = self.eval_context_mut();

        let host_socket = socket.socket.borrow();

        if dest_addr.is_some() && socket.has_default_destination.get() {
            // The sendto POSIX man page specifies that "[...] If a peer address has been pre-specified, either
            // the message shall be sent to the address specified by dest_addr (overriding the pre-specified
            // peer address), or the function shall return -1 and set errno to [EISCONN]. [...]".
            // It turns out that Linux implements the prior option while BSD-based platforms implement the latter.
            // Since both are allowed by POSIX, we try to model this as good as possible.

            if matches!(ecx.tcx.sess.target.os, Os::FreeBsd | Os::MacOs) {
                // We're on a BSD target. When both `dest_addr` and a default destination address
                // are available, those platforms return EISCONN and don't perform the operation.
                return interp_ok(Err(LibcError("EISCONN")));
            } else if cfg!(any(target_os = "freebsd", target_vendor = "apple")) {
                // We're on a BSD host and on a non-BSD target. On such targets `dest_addr` should
                // overwrite the default destination address, however, we cannot achieve this nicely
                // since the host disallows that. We thus throw an unsupported error.
                throw_unsup_format!(
                    "sendto: overwriting default destination address is not supported for UDP sockets on BSD hosts"
                )
            }
        }

        // This is a *non-blocking* write.
        let result = ecx.write_to_host(buffer_ptr, length, |bytes| {
            match dest_addr {
                Some(dest_addr) => host_socket.send_to(bytes, dest_addr),
                None => host_socket.send(bytes),
            }
        })?;

        if result.is_ok() {
            // A write should never succeed when the `write_closed` readiness is set for this socket.
            assert!(!socket.io_readiness.borrow().write_closed, "successful write after close");

            if dest_addr.is_some() {
                // After the first successful `sendto`, the socket gets implicitly bound to a local address.
                socket.is_bound.set(true);
            } else {
                // Can only succeed if `connect` was called, which binds the socket.
                assert!(socket.is_bound.get());
            }
        }

        match result {
            Err(IoError::HostError(e)) if matches!(e.kind(), io::ErrorKind::WouldBlock) => {
                // We know that the source is not writable so we need to update its readiness.
                socket.io_readiness.borrow_mut().writable = false;
                ecx.update_fd_readiness(socket.clone(), ReadinessUpdateFlags::DEFAULT)?;

                interp_ok(Err(IoError::HostError(e)))
            }
            Err(IoError::HostError(e))
                if cfg!(windows) && matches!(e.kind(), io::ErrorKind::NotConnected) =>
            {
                // Windows returns ENOTCONN instead of EDESTADDRREQ when attempting to `send` without
                // having a default destination address set. We thus remap this error to an EDESTADDRREQ.
                interp_ok(Err(IoError::LibcError("EDESTADDRREQ")))
            }
            Err(IoError::HostError(e))
                if cfg!(windows)
                    && e.raw_os_error().is_some_and(|errno| errno == /* WSAEMSGSIZE */ 10040) =>
            {
                // We attempted to write a message which doesn't fit into a single UDP datagram.
                // On Windows hosts this returns WSAEMSGSIZE while on Unix-like hosts this returns EMSGSIZE.
                // Since WSAEMSGSIZE doesn't map to a `io::ErrorKind`, Miri throws an unsupported error if we
                // would just return this I/O error. We thus manually remap this error to EMSGSIZE on Windows
                // hosts. The mapping needs to be performed here and not directly inside the callback function
                // where the `host_socket.send` occurs since this callback function returns a result with `io::Error`
                // instead of `IoError`.
                // FIXME: remove this if WSAEMSGSIZE ever maps to a `io::ErrorKind`.
                interp_ok(Err(IoError::LibcError("EMSGSIZE")))
            }
            result => interp_ok(result),
        }
    }

    /// Block the thread until we can receive bytes from the socket or an error occurred.
    ///
    /// This recursively calls itself should the operation still block for some reason.
    fn block_for_recv(
        &mut self,
        socket: FileDescriptionRef<UdpSocket>,
        deadline: Option<Deadline>,
        buffer_ptr: Pointer,
        length: u64,
        should_peek: bool,
        finish: DynMachineCallback<'tcx, Result<(u64, SocketAddr), IoError>>,
    ) -> InterpResult<'tcx> {
        let recv = self.eval_context_mut();
        // Since the callback holds a strong reference to the socket, the file description
        // won't be closed as long as some thread is blocked on it. While this reflects
        // what Linux does, for other Unix systems this might differ from the native behavior.
        recv.block_thread_for_io(
            socket.clone(),
            BlockingIoInterest::Read,
            deadline.clone(),
            callback!(@capture<'tcx> {
                socket: FileDescriptionRef<UdpSocket>,
                deadline: Option<Deadline>,
                buffer_ptr: Pointer,
                length: u64,
                should_peek: bool,
                finish: DynMachineCallback<'tcx, Result<(u64, SocketAddr), IoError>>,
            } |ecx, kind: UnblockKind| {
                // Remove the blocking I/O interest for unblocking this thread.
                ecx.machine.blocking_io.remove_blocked_thread(socket.id(), ecx.machine.threads.active_thread());

                match kind {
                    UnblockKind::Ready => { /* fall-through to below */ },
                    // When the read timeout is exceeded EAGAIN/EWOULDBLOCK is returned.
                    UnblockKind::TimedOut => return finish.call(ecx, Err(LibcError("EWOULDBLOCK")))
                }

                match ecx.try_non_block_recv(&socket, buffer_ptr, length, should_peek)? {
                    Err(IoError::HostError(e)) if e.kind() == io::ErrorKind::WouldBlock => {
                        // We need to block the thread again as it would still block.
                        ecx.block_for_recv(socket, deadline, buffer_ptr, length, should_peek, finish)
                    },
                    result => finish.call(ecx, result)
                }
            }),
        )
    }

    /// Attempt to receive bytes from the socket in a non-blocking manner.
    fn try_non_block_recv(
        &mut self,
        socket: &FileDescriptionRef<UdpSocket>,
        buffer_ptr: Pointer,
        length: u64,
        should_peek: bool,
    ) -> InterpResult<'tcx, Result<(u64, SocketAddr), IoError>> {
        let ecx = self.eval_context_mut();

        let host_socket = socket.socket.borrow();

        // If the `read_from_host` below succeeds, this contains the address from where
        // the data originated. Otherwise, this is `None`.
        let mut origin_addr = None;

        let mut recv_from_host_socket = |buf: &mut [u8]| {
            let result =
                if should_peek { host_socket.peek_from(buf) } else { host_socket.recv_from(buf) };

            result.map(|(bytes, addr)| {
                // Store the origin address in `origin_addr`. We need to do this since
                // `read_from_host` expects the callback to return `io::Result<usize>`.
                origin_addr.replace(addr);

                // Return the amount of bytes received.
                bytes
            })
        };

        // This is a *non-blocking* read/peek.
        let result = ecx.read_from_host(buffer_ptr, length, |buf| {
            if cfg!(windows) {
                // When trying to read a message from the socket which does not fit our
                // provided buffer, POSIX specifies that the provided buffer should be
                // filled and the remaining bytes should be discarded. The receive operation
                // should then return the amount of bytes read. Windows hosts fill the buffer
                // and discard the excess bytes but return WSAEMSGSIZE instead of the bytes read.
                // Since the return type of the corresponding operations in the standard library
                // (and thus also in mio) is `io::Result<(usize, SocketAddr)>`, we essentially
                // "lose" the address from where this data originated.
                // To circumvent this, we use a temporary 64KiB buffer which fits any UDP
                // datagram, receive into this temporary buffer, and then copy at most the first
                // `length` bytes from the temporary buffer into `buf`.

                #[expect(clippy::as_conversions)]
                let mut tmp_buf = [0u8; u16::MAX as usize];
                let result = recv_from_host_socket(&mut tmp_buf).map(|bytes| {
                    let bytes_received = bytes.min(usize::try_from(length).unwrap());
                    buf[..bytes_received].copy_from_slice(&tmp_buf[..bytes_received]);

                    // Return the actual amount of bytes written to `buf`.
                    bytes_received
                });

                if let Err(ref e) = result {
                    // Since we received into a big enough temporary buffer, we should never
                    // fail with WSAEMSGSIZE.
                    assert_ne!(e.raw_os_error(), Some(/* WSAEMSGSIZE */ 10040))
                }

                return result;
            }

            recv_from_host_socket(buf)
        })?;

        match result {
            Err(IoError::HostError(e)) if matches!(e.kind(), io::ErrorKind::WouldBlock) => {
                // We know that the source is not readable so we need to update its readiness.
                socket.io_readiness.borrow_mut().readable = false;
                ecx.update_fd_readiness(socket.clone(), ReadinessUpdateFlags::DEFAULT)?;

                interp_ok(Err(IoError::HostError(e)))
            }
            // Since `origin_addr` is only `Some` when `result` is `Ok`, it's okay to unwrap
            // `origin_addr` in the mapping function of `result`.
            result => interp_ok(result.map(|bytes| (bytes, origin_addr.unwrap()))),
        }
    }
}

impl SourceFileDescription for UdpSocket {
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()> {
        let mut socket = self.socket.borrow_mut();
        f(&mut *socket)
    }

    fn get_readiness_mut(&self) -> RefMut<'_, Readiness> {
        self.io_readiness.borrow_mut()
    }
}
