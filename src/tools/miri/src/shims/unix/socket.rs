use std::net::{Shutdown, SocketAddr};

use rustc_abi::Size;
use rustc_target::spec::Os;

use crate::shims::FileDescriptionRef;
use crate::shims::files::FdNum;
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::socket_address::EvalContextExt as _;
use crate::shims::unix::tcp_socket::TcpSocket;
use crate::*;

#[derive(Debug, PartialEq)]
pub enum SocketFamily {
    // IPv4 internet protocols
    IPv4,
    // IPv6 internet protocols
    IPv6,
}

/// Represents unix-specific socket file descriptions.
///
/// Not to be confused with Unix domain sockets.
pub trait UnixSocketFileDescription: UnixFileDescription {
    /// Bind the socket to `address`.
    fn bind<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _address: SocketAddr,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        throw_unsup_format!("cannot bind {}", self.name());
    }

    /// Start listening on the socket.
    /// `backlog` specifies how many pending incoming connections can exist
    /// at the same time before new requests are rejected.
    fn listen<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _backlog: i32,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        throw_unsup_format!("cannot listen on {}", self.name());
    }

    /// Accept an incoming connection on the socket.
    /// `is_client_sock_non_block` specifies whether the newly accepted
    /// client connection should be non-blocking.
    /// After a successful accept, `finish` should be called with a
    /// tuple containing the file descriptor of the peer socket and
    /// it's address.
    fn accept<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _is_client_sock_non_block: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<(FdNum, SocketAddr), IoError>>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot accept {}", self.name());
    }

    /// Connect the socket to `address`.
    fn connect<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _address: SocketAddr,
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<(), IoError>>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot connect {}", self.name());
    }

    /// Recieve data on the socket into the given buffer `ptr`.
    /// `len` indicates how many bytes we should try to receive.
    /// `is_peek` specifies whether the receive removes the bytes
    /// from the receive buffer ([`false`]) or leaves them in the
    /// reiceve buffer ([`true`]).
    /// `is_non_block` specifies whether the receive operation is
    /// non-blocking.
    /// After a successful receive, `finish` should be called with
    /// the amount of bytes received.
    fn recv<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _ptr: Pointer,
        _len: usize,
        _is_peek: bool,
        _is_non_block: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot receive from {}", self.name());
    }

    /// Send data from the given buffer `ptr` into the socket.
    /// `len` indicates how many bytes we should try to send.
    /// `is_non_block` specifies whether the send operation is
    /// non-blocking.
    /// After a successful send, `finish` should be called with
    /// the amount of bytes sent.
    fn send<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _ptr: Pointer,
        _len: usize,
        _is_non_block: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot send to {}", self.name());
    }

    /// Set the socket option `option` on `level`.
    /// `value_ptr` points to the new value of the socket option,
    /// and `value_len` contains the amount of bytes the value uses
    /// at `value_ptr`.
    fn setsockopt<'tcx>(
        self: FileDescriptionRef<Self>,
        _level: i32,
        _option: i32,
        _value_ptr: Pointer,
        _value_len: u64,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        throw_unsup_format!("cannot set socket option on {}", self.name());
    }

    /// Get the value of a socket option `option` on `level`.
    fn getsockopt<'tcx>(
        self: FileDescriptionRef<Self>,
        _level: i32,
        _option: i32,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<MPlaceTy<'tcx>, IoError>> {
        throw_unsup_format!("cannot get socket option for {}", self.name());
    }

    /// Get the address of the socket.
    fn getsockname<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<SocketAddr, IoError>> {
        throw_unsup_format!("cannot get socket name for {}", self.name());
    }

    /// Get the address of the connected socket.
    fn getpeername<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
        _finish: DynMachineCallback<'tcx, Result<SocketAddr, IoError>>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("cannot get peer name for {}", self.name());
    }

    /// Shut down the read and/or the write end of the socket.
    fn shutdown<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        _how: Shutdown,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        throw_unsup_format!("cannot shut down {}", self.name());
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
            return this.set_errno_and_return_neg1_i32(LibcError("EACCES"));
        }

        let mut is_non_block = false;

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
                is_non_block = true;
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
        let fd = fds.new_ref(TcpSocket::new(family, is_non_block));

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
            Err(e) => return this.set_errno_and_return_neg1_i32(e),
        };

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };

        let socket = fd.as_unix(this).as_socket(this);
        match socket.bind(this.machine.communicate(), address, this)? {
            Ok(_) => interp_ok(Scalar::from_i32(0)),
            Err(e) => this.set_errno_and_return_neg1_i32(e),
        }
    }

    fn listen(&mut self, socket: &OpTy<'tcx>, backlog: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let socket = this.read_scalar(socket)?.to_i32()?;
        let backlog = this.read_scalar(backlog)?.to_i32()?;

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };

        let socket = fd.as_unix(this).as_socket(this);
        match socket.listen(this.machine.communicate(), backlog, this)? {
            Ok(_) => interp_ok(Scalar::from_i32(0)),
            Err(e) => this.set_errno_and_return_neg1_i32(e),
        }
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
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        let socket = fd.as_unix(this).as_socket(this);

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

        let dest = dest.clone();
        socket.accept(
            this.machine.communicate(),
            is_client_sock_nonblock,
            this,
            callback!(
                @capture<'tcx> {
                    address_ptr: Pointer,
                    address_len_ptr: Pointer,
                    dest: MPlaceTy<'tcx>
                } |this, result: Result<(i32, SocketAddr), IoError>| {
                    let (client_sockfd, address) = match result {
                        Ok((sockfd, address)) => (sockfd, address),
                        Err(e) => return this.set_errno_and_return_neg1(e, &dest),
                    };

                    if address_ptr != Pointer::null() {
                        // We only attempt a write if the address pointer is not a null pointer.
                        // If the address pointer is a null pointer the user isn't interested in the
                        // address and we don't need to write anything.
                        this.write_socket_address(&address, address_ptr, address_len_ptr, "accept4")?;
                    }

                    // We need to create the scalar using the destination size since
                    // `syscall(SYS_accept4, ...)` returns a long which doesn't match
                    // the int returned from the `accept`/`accept4` syscalls.
                    // See <https://man7.org/linux/man-pages/man2/syscall.2.html>.
                    this.write_scalar(Scalar::from_int(client_sockfd, dest.layout.size), &dest)
                }
            ),
        )
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
            Err(e) => return this.set_errno_and_return_neg1(e, dest),
        };

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        let socket = fd.as_unix(this).as_socket(this);

        let dest = dest.clone();

        socket.connect(
            this.machine.communicate(),
            address,
            this,
            callback!(
                @capture<'tcx> {
                    dest: MPlaceTy<'tcx>
                 } |this, result: Result<(), IoError>| {
                     match result {
                         Ok(_) => this.write_null(&dest),
                         Err(e) => this.set_errno_and_return_neg1(e, &dest)
                     }
                 }
            ),
        )
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
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        let socket = fd.as_unix(this).as_socket(this);

        let mut is_non_block = false;

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
                is_non_block = true;
            }
        }

        if flags != 0 {
            throw_unsup_format!(
                "send: flag {flags:#x} is unsupported, only MSG_NOSIGNAL and MSG_DONTWAIT are allowed",
            );
        }

        let dest = dest.clone();

        socket.send(
            this.machine.communicate(),
            buffer_ptr,
            length,
            is_non_block,
            this,
            callback!(
                @capture<'tcx> {
                    dest: MPlaceTy<'tcx>,
                } |this, result: Result<usize, IoError>| {
                    match result {
                        Ok(bytes_sent) =>
                            this.write_scalar(Scalar::from_target_usize(bytes_sent.try_into().unwrap(), this), &dest),
                        Err(e) => this.set_errno_and_return_neg1(e, &dest)
                    }
                }
            )
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
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        let socket = fd.as_unix(this).as_socket(this);

        let mut is_peek = false;
        let mut is_non_block = false;

        // Interpret the flag. Every flag we recognize is "subtracted" from `flags`, so
        // if there is anything left at the end, that's an unsupported flag.

        let msg_peek = this.eval_libc_i32("MSG_PEEK");
        if flags & msg_peek == msg_peek {
            is_peek = true;
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
                is_non_block = true;
            }
        }

        if flags != 0 {
            throw_unsup_format!(
                "recv: flag {flags:#x} is unsupported, only MSG_PEEK, MSG_DONTWAIT \
                and MSG_CMSG_CLOEXEC are allowed",
            );
        }

        let dest = dest.clone();

        socket.recv(
            this.machine.communicate(),
            buffer_ptr,
            length,
            is_peek,
            is_non_block,
            this,
            callback!(
                @capture<'tcx> {
                    dest: MPlaceTy<'tcx>,
                } |this, result: Result<usize, IoError>| {
                    match result {
                        Ok(bytes_sent) =>
                            this.write_scalar(Scalar::from_target_usize(bytes_sent.try_into().unwrap(), this), &dest),
                        Err(e) => this.set_errno_and_return_neg1(e, &dest)
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
        let option_value_ptr = this.read_pointer(option_value)?;
        let socklen_layout = this.libc_ty_layout("socklen_t");
        let option_len: u64 =
            this.read_scalar(option_len)?.to_int(socklen_layout.size)?.try_into().unwrap();

        // Get the file handle
        let Some(fd) = this.machine.fds.get(socket) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };

        let socket = fd.as_unix(this).as_socket(this);
        let result = socket.setsockopt(level, option_name, option_value_ptr, option_len, this)?;
        match result {
            Ok(_) => interp_ok(Scalar::from_i32(0)),
            Err(e) => this.set_errno_and_return_neg1_i32(e),
        }
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
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };

        let socket = fd.as_unix(this).as_socket(this);

        if option_value_ptr == Pointer::null() || option_len_ptr == Pointer::null() {
            // This socket option returns a value and thus we need to return EFAULT
            // when either the value or the length pointers are null pointers.
            return this.set_errno_and_return_neg1_i32(LibcError("EFAULT"));
        }

        let socklen_layout = this.libc_ty_layout("socklen_t");
        let option_len_ptr_mplace = this.ptr_to_mplace(option_len_ptr, socklen_layout);
        let option_len: usize = this
            .read_scalar(&option_len_ptr_mplace)?
            .to_int(socklen_layout.size)?
            .try_into()
            .unwrap();

        // `socket.getsockopt` returns a temporary buffer as `option_value_ptr` might not point
        // to a large enough buffer, in which case we have to truncate.
        let value_mplace = match socket.getsockopt(level, option_name, this)? {
            Ok(value_mplace) => value_mplace,
            Err(e) => return this.set_errno_and_return_neg1_i32(e),
        };

        // Truncated size of the output value.
        let output_value_len = value_mplace.layout.size.min(Size::from_bytes(option_len));
        // Copy the truncated value into the buffer pointed to by `option_value_ptr`.
        this.mem_copy(
            value_mplace.ptr(),
            option_value_ptr,
            // Truncate the value to fit the provided buffer.
            output_value_len,
            // The buffers are guaranteed to not overlap since the `value_mplace`
            // was just newly allocated on the stack.
            true,
        )?;
        // Deallocate the value buffer as it was only needed to store the value and
        // copy it into the buffer pointed to by `option_value_ptr`.
        this.deallocate_ptr(value_mplace.ptr(), None, MemoryKind::Stack)?;

        // On output, the length pointer contains the amount of bytes written -- not the size
        // of the value before truncation.
        this.write_scalar(
            Scalar::from_uint(output_value_len.bytes(), socklen_layout.size),
            &option_len_ptr_mplace,
        )?;

        interp_ok(Scalar::from_i32(0))
    }
}
