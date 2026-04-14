//! Userspace helper library for writing VFS providers and channel services.
//!
//! This library provides ergonomic wrappers around the raw Thing-OS IPC
//! syscalls, making it easy to write drivers and system services without
//! dealing with framing details manually.
//!
//! # Modules
//!
//! - [`channel`] — channel send/recv wrappers with RPC framing support
//! - [`provider`] — VFS provider server loop
//! - [`rpc`] — typed request/reply client and server helpers
//!
//! # Quick Start
//!
//! **Writing a simple service** (see [`channel`]):
//!
//! ```ignore
//! use ipc_helpers::rpc::{RpcServer, RpcRequest};
//!
//! let (write_h, read_h) = stem::syscall::channel::channel_create(4096).unwrap();
//! // publish write_h to clients …
//!
//! let mut server = RpcServer::new(read_h);
//! loop {
//!     let req = server.next().unwrap();
//!     server.reply(req.request_id, b"pong").unwrap();
//! }
//! ```
//!
//! **Writing a VFS provider** (see [`provider`]):
//!
//! ```ignore
//! use ipc_helpers::provider::{ProviderLoop, ProviderResponse};
//! use abi::vfs_rpc::VfsRpcOp;
//! use abi::errors::Errno;
//!
//! let mut lp = ProviderLoop::new(vfs_read);
//! loop {
//!     let req = lp.next_request().unwrap();
//!     let resp = match req.op {
//!         VfsRpcOp::Lookup => ProviderResponse::ok_u64(1),
//!         VfsRpcOp::Read => ProviderResponse::ok_bytes(b"hello\n"),
//!         VfsRpcOp::Stat => ProviderResponse::ok_stat(0o100644, 6, 1),
//!         VfsRpcOp::Close => ProviderResponse::ok_empty(),
//!         _ => ProviderResponse::err(Errno::ENOSYS),
//!     };
//!     lp.send_response(req.resp_port, resp).unwrap();
//! }
//! ```

#![no_std]
extern crate alloc;

pub mod channel;
pub mod provider;
pub mod rpc;
