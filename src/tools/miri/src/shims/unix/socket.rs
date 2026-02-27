use std::cell::{Cell, RefCell};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6, TcpListener};

use rustc_abi::Size;
use rustc_const_eval::interpret::{InterpResult, interp_ok};
use rustc_middle::throw_unsup_format;
use rustc_target::spec::Os;

use crate::diagnostics::SpanDedupDiagnostic;
use crate::shims::files::{FdId, FileDescription};
use crate::{OpTy, Scalar, *};

/// Backlog value passed to the `listen` syscall by the standard library
const SUPPORTED_LISTEN_BACKLOG: i32 = 128;

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
    #[expect(unused)]
    Listening(TcpListener),
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
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, std::io::Result<()>>
    where
        Self: Sized,
    {
        interp_ok(Ok(()))
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
            // Solaris, and Illumos targets
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
            return interp_ok(this.eval_libc("EBADF"));
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return interp_ok(this.eval_libc("ENOTSOCK"));
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
                        // Linux man page states that `EINVAL` is used when there is an addres family mismatch.
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
        let backlog = this.read_scalar(backlog)?.to_i32()?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return interp_ok(this.eval_libc("EBADF"));
        };

        let Some(socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return interp_ok(this.eval_libc("ENOTSOCK"));
        };

        assert!(this.machine.communicate(), "cannot have `Socket` with isolation enabled!");

        // Only allow the same backlog value as the standard library uses since the standard library
        // doesn't provide a way to set a custom value.
        if backlog != SUPPORTED_LISTEN_BACKLOG {
            // The first time this happens at a particular location, print a warning.
            static DEDUP: SpanDedupDiagnostic = SpanDedupDiagnostic::new();
            this.dedup_diagnostic(&DEDUP, |first| {
                NonHaltingDiagnostic::SocketListenUnsupportedBacklog {
                    details: first,
                    provided: backlog,
                    supported: SUPPORTED_LISTEN_BACKLOG,
                }
            });
        }

        let mut state = socket.state.borrow_mut();

        match &*state {
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
        }

        interp_ok(Scalar::from_i32(0))
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
            return interp_ok(this.eval_libc("EBADF"));
        };

        let Some(_socket) = fd.downcast::<Socket>() else {
            // Man page specifies to return ENOTSOCK if `fd` is not a socket.
            return interp_ok(this.eval_libc("ENOTSOCK"));
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
}
