use std::cell::Cell;
use std::io;

use crate::shims::files::{FileDescription, FileDescriptionRef};
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::socket::UnixSocketFileDescription;
use crate::*;

#[derive(Debug)]
#[expect(unused)]
pub(super) struct UdpSocket {
    /// Family of the socket, used to ensure that the socket only binds/connects to addresses
    /// of the same family.
    family: socket2::Domain,
    /// The underlying host socket.
    socket: mio::net::UdpSocket,
    /// Whether this fd is non-blocking or not.
    is_non_block: Cell<bool>,
}

impl UdpSocket {
    pub fn new(family: socket2::Domain, is_non_block: bool) -> io::Result<Self> {
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

        // TODO: Add the underlying host socket to the blocking I/O manager.

        Ok(UdpSocket { family, socket, is_non_block: Cell::new(is_non_block) })
    }
}

impl FileDescription for UdpSocket {
    fn name(&self) -> &'static str {
        "udp socket"
    }

    fn as_unix(self: FileDescriptionRef<Self>) -> FileDescriptionRef<dyn UnixFileDescription> {
        self
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

impl UnixSocketFileDescription for UdpSocket {}
