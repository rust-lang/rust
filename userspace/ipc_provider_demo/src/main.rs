//! ipc_provider_demo — IPC Cookbook Recipe 4: minimal VFS provider.
//!
//! This program demonstrates how to expose a virtual filesystem tree under a
//! mount point using the Thing-OS VFS provider protocol.
//!
//! # What this shows
//!
//! 1. **Channel creation** — create the provider channel pair.
//! 2. **Mount** — register the channel write-end with the kernel via
//!    `vfs_mount(provider_write_handle, path)`.
//! 3. **Provider loop** — use [`ProviderLoop`] to read typed kernel requests
//!    and dispatch them with [`ProviderResponse`] values.
//!
//! # Virtual filesystem exposed
//!
//! Mount point: `/run/cookbook`
//!
//! ```text
//! /run/cookbook/
//!   hello.txt   — contains "Hello from the VFS provider!\n"
//! ```
//!
//! # Running
//!
//! ```text
//! # In the Thing-OS shell:
//! $ ipc_provider_demo &
//! $ cat /run/cookbook/hello.txt
//! Hello from the VFS provider!
//! ```
//!
//! # See Also
//!
//! `docs/concepts/ipc_cookbook.md` Recipe 4 — VFS provider implementation.
//! `libs/ipc_helpers/src/provider.rs` — ProviderLoop helper.
//! `userspace/iso9660d/` — full reference implementation.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use abi::errors::Errno;
use abi::vfs_rpc::VfsRpcOp;
use ipc_helpers::provider::{ProviderLoop, ProviderResponse};
use stem::syscall::{channel_create, vfs_mount};
use stem::{info, warn};

/// Mount point for this provider.
const MOUNT_POINT: &str = "/run/cookbook";

/// The single file this provider exposes.
const FILE_NAME: &[u8] = b"hello.txt";

/// Synthetic handle value for hello.txt (any non-zero u64 is fine).
const HELLO_HANDLE: u64 = 1;

/// Content served by READ.
const HELLO_CONTENT: &[u8] = b"Hello from the VFS provider!\n";

#[stem::main]
fn main(_arg: usize) -> ! {
    info!("ipc_provider_demo: starting up");

    // ── 1. Create the provider channel pair ──────────────────────────────
    //
    // channel_create returns (write_handle, read_handle).
    // The kernel sends VFS RPC requests to the write-handle; we read them
    // from the read-handle.
    let (write_h, read_h) = match channel_create(65536) {
        Ok(pair) => pair,
        Err(e) => {
            warn!("ipc_provider_demo: channel_create failed: {:?}", e);
            stem::syscall::exit(1);
        }
    };
    info!(
        "ipc_provider_demo: channel pair write_h={} read_h={}",
        write_h, read_h
    );

    // ── 2. Mount the provider at MOUNT_POINT ─────────────────────────────
    //
    // The kernel accepts the write-handle and routes all VFS operations
    // under the mount point through it.
    match vfs_mount(write_h, MOUNT_POINT) {
        Ok(()) => info!("ipc_provider_demo: mounted at {}", MOUNT_POINT),
        Err(e) => {
            warn!(
                "ipc_provider_demo: vfs_mount failed: {:?} (continuing anyway for demo)",
                e
            );
            // Continue so the provider loop still demonstrates the dispatch logic.
        }
    }

    // ── 3. Provider loop ──────────────────────────────────────────────────
    //
    // ProviderLoop reads raw VFS RPC frames from the channel and decodes
    // the header for us.  We only need to return a ProviderResponse for
    // each request.
    let mut lp = ProviderLoop::new(read_h);
    let mut request_count: u64 = 0;

    loop {
        let req = match lp.next_request() {
            Ok(r) => r,
            Err(e) => {
                info!(
                    "ipc_provider_demo: channel closed ({:?}) after {} requests — exiting",
                    e, request_count
                );
                break;
            }
        };

        request_count += 1;
        info!(
            "ipc_provider_demo: request #{} op={:?}",
            request_count, req.op
        );

        let resp = dispatch(&req.op, &req.payload);
        if let Err(e) = lp.send_response(req.resp_port, resp) {
            warn!("ipc_provider_demo: send_response failed: {:?}", e);
        }
    }

    info!("ipc_provider_demo: clean shutdown");
    stem::syscall::exit(0);
}

/// Dispatch a single VFS RPC operation and return the appropriate response.
fn dispatch(op: &VfsRpcOp, payload: &[u8]) -> ProviderResponse {
    match op {
        // Lookup resolves a path component to a handle.
        // Payload layout: 4-byte parent handle (u32) + path bytes.
        VfsRpcOp::Lookup => {
            let path = if payload.len() > 4 {
                core::str::from_utf8(&payload[4..]).unwrap_or("")
            } else {
                ""
            };
            if path.as_bytes() == FILE_NAME {
                ProviderResponse::ok_u64(HELLO_HANDLE)
            } else {
                ProviderResponse::err(Errno::ENOENT)
            }
        }

        // Read returns file content.
        // Payload layout: 8-byte handle + 8-byte offset + 4-byte length.
        VfsRpcOp::Read => {
            if payload.len() >= 8 {
                let handle = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0; 8]));
                if handle == HELLO_HANDLE {
                    return ProviderResponse::ok_read(HELLO_CONTENT);
                }
            }
            ProviderResponse::err(Errno::EBADF)
        }

        // Stat returns mode, size, inode number.
        VfsRpcOp::Stat => {
            if payload.len() >= 8 {
                let handle = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0; 8]));
                if handle == HELLO_HANDLE {
                    // regular file, world-readable, size = len(HELLO_CONTENT)
                    return ProviderResponse::ok_stat(
                        0o100_444,
                        HELLO_CONTENT.len() as u64,
                        HELLO_HANDLE,
                    );
                }
            }
            ProviderResponse::err(Errno::EBADF)
        }

        // Readdir lists directory entries.
        // For the root handle (0) we return a single entry for hello.txt.
        VfsRpcOp::Readdir => {
            let handle = if payload.len() >= 8 {
                u64::from_le_bytes(payload[..8].try_into().unwrap_or([0; 8]))
            } else {
                0
            };
            if handle == 0 {
                // Simple dirent: inode(u64) + name_len(u8) + name bytes
                let mut entry = alloc::vec![0u8; 8 + 1 + FILE_NAME.len()];
                entry[..8].copy_from_slice(&HELLO_HANDLE.to_le_bytes());
                entry[8] = FILE_NAME.len() as u8;
                entry[9..].copy_from_slice(FILE_NAME);
                ProviderResponse::ok_read(&entry)
            } else {
                // Non-root handles are not directories.
                ProviderResponse::ok_read(&[])
            }
        }

        // Close is a no-op (handles are stateless).
        VfsRpcOp::Close => ProviderResponse::ok_empty(),

        // Write is not supported.
        VfsRpcOp::Write => ProviderResponse::err(Errno::EROFS),

        // Poll always reports readable.
        VfsRpcOp::Poll => ProviderResponse::ok_bytes(&[1u8]),

        // All other operations are not implemented.
        _ => ProviderResponse::err(Errno::ENOSYS),
    }
}
