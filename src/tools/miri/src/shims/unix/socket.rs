use std::net::{Shutdown, SocketAddr};

use crate::shims::FileDescriptionRef;
use crate::shims::files::FdNum;
use crate::shims::unix::UnixFileDescription;
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
