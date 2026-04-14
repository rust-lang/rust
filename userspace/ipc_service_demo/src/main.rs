//! ipc_service_demo — demonstrates the happy path for writing a channel service
//! using [`ipc_helpers`].
//!
//! # What this shows
//!
//! 1. **Startup handshake** — the service creates a channel pair, publishes the
//!    write end so clients can find it, then waits for the first client message.
//! 2. **Event loop** — uses [`RpcServer`] to receive typed requests and
//!    [`RpcServer::reply`] to send responses without touching raw byte buffers.
//! 3. **Clean shutdown** — when [`RpcServer::next`] returns `Err`, the peer
//!    has closed the channel; the service logs and exits gracefully.
//!
//! # Running
//!
//! This binary is intended to run under the Thing-OS userspace.  Build it as
//! part of the workspace and include it in the OS image.  It publishes its
//! write handle at `/services/echo` so any program can send it an RPC request
//! and receive an echo reply.
//!
//! ```text
//! # Inside QEMU / the OS shell:
//! $ ipc_service_demo &   # start the echo service
//! $ ipc_echo_client      # optional test client (not included here)
//! ```
//!
//! # Boilerplate comparison
//!
//! Without `ipc_helpers`, a service loop requires ~40 lines of header encoding,
//! EAGAIN retry loops, and manual buffer management.  With `ipc_helpers`:
//!
//! ```ignore
//! let mut server = RpcServer::new(read_h);
//! loop {
//!     let req = server.next()?;         // blocks; auto-decodes RPC header
//!     server.reply(req.request_id, write_h, &req.payload)?;  // echo
//! }
//! ```
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use abi::syscall::vfs_flags::{O_CREAT, O_WRONLY};
use ipc_helpers::channel::OwnedChannel;
use ipc_helpers::rpc::RpcServer;
use stem::syscall::channel::channel_create;
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_write};
use stem::{info, warn};

/// Path where the service publishes its write handle.
const SERVICE_PATH: &str = "/services/echo";

#[stem::main]
fn main(_arg: usize) -> ! {
    info!("ipc_service_demo: starting up");

    // ── 1. Startup: create channel pair ──────────────────────────────────────
    //
    // The kernel returns (write_handle, read_handle).  We keep the read end
    // to receive requests and publish the write end so clients can send to us.
    let (write_h, read_h) = match channel_create(65536) {
        Ok(pair) => pair,
        Err(e) => {
            warn!("ipc_service_demo: channel_create failed: {:?}", e);
            loop {
                stem::yield_now();
            }
        }
    };

    // Wrap the read end in OwnedChannel for automatic close on drop.
    let _read_channel = OwnedChannel::new(read_h);

    // ── 2. Startup handshake: publish write handle at a known VFS path ────────
    //
    // Write the handle number as a decimal string into the services directory.
    // Clients open this path, read the handle number, and use it to connect.
    let handle_str = alloc::format!("{}\n", write_h);
    match vfs_open(SERVICE_PATH, O_WRONLY | O_CREAT) {
        Ok(fd) => {
            if let Err(e) = vfs_write(fd, handle_str.as_bytes()) {
                warn!("ipc_service_demo: failed to write handle: {:?}", e);
            } else {
                info!(
                    "ipc_service_demo: published write_h={} at {}",
                    write_h, SERVICE_PATH
                );
            }
            let _ = vfs_close(fd);
        }
        Err(e) => {
            warn!(
                "ipc_service_demo: failed to open {}: {:?}",
                SERVICE_PATH, e
            );
            // Continue anyway — the channel still works; clients just need
            // to know the handle number out-of-band.
        }
    }

    info!("ipc_service_demo: entering RPC event loop (read_h={})", read_h);

    // ── 3. Event loop: serve requests ─────────────────────────────────────────
    //
    // RpcServer::next() blocks until a complete framed request arrives, then
    // decodes the RpcHeader automatically.  We echo the payload back to the
    // client as a demonstration.
    let mut server = RpcServer::new(read_h);
    let mut request_count: u64 = 0;

    loop {
        let req = match server.next() {
            Ok(r) => r,
            // ── 4. Clean shutdown ─────────────────────────────────────────────
            //
            // EPIPE / ENOTCONN means the peer closed the channel.  Exit the
            // event loop and let the process terminate cleanly.
            Err(e) => {
                info!(
                    "ipc_service_demo: channel closed ({:?}) after {} requests — shutting down",
                    e, request_count
                );
                break;
            }
        };

        request_count += 1;
        info!(
            "ipc_service_demo: request #{} id={} payload_len={}",
            request_count,
            req.request_id,
            req.payload.len()
        );

        // Echo the request payload back as the reply.
        if let Err(e) = server.reply(req.request_id, write_h, &req.payload) {
            warn!("ipc_service_demo: reply failed: {:?}", e);
        }
    }

    info!("ipc_service_demo: clean shutdown complete");
    stem::syscall::exit(0);
}
