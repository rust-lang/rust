use std::cell::{Cell, RefCell, RefMut};
use std::io;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use std::time::Duration;

use mio::event::Source;
use rustc_target::spec::Os;

use crate::shims::files::{FileDescription, FileDescriptionRef};
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::socket::{EvalContextExt, UnixSocketFileDescription};
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

        if let Err(e) = self.socket.borrow().connect(address) {
            return finish.call(ecx, Err(IoError::HostError(e)));
        }

        // POSIX specifies that unbound sockets get bound to an unassigned
        // local address after `connect` has been invoked on them.
        // We thus treat the socket as implicitly bound.
        self.is_bound.set(true);

        finish.call(ecx, Ok(()))
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

impl SourceFileDescription for UdpSocket {
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()> {
        let mut socket = self.socket.borrow_mut();
        f(&mut *socket)
    }

    fn get_readiness_mut(&self) -> RefMut<'_, Readiness> {
        self.io_readiness.borrow_mut()
    }
}
