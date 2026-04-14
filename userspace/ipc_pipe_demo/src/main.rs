//! ipc_pipe_demo — IPC Cookbook Recipe 1: parent/child stdio via pipe.
//!
//! This program demonstrates the core pipe primitives:
//!
//! 1. **Create a pipe** — `pipe()` returns a `[read_fd, write_fd]` pair.
//! 2. **Write to write-end** — data buffered in the kernel.
//! 3. **Read from read-end** — data arrives in order (FIFO).
//! 4. **EOF detection** — when the write-end is closed, reads return 0.
//!
//! In a multi-process scenario the parent keeps the read end and the child
//! inherits the write end (passed via `spawn_process_ex` stdout piping or
//! an env var carrying the fd number).  This single-process demo shows the
//! same round-trip path in isolation so it can be understood without a
//! second binary.
//!
//! # Running
//!
//! ```text
//! # In the Thing-OS shell:
//! $ ipc_pipe_demo
//! ```
//!
//! # See Also
//!
//! `docs/concepts/ipc_cookbook.md` Recipe 1 — parent/child stdio via pipe.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use stem::syscall::vfs::{pipe, vfs_close, vfs_read, vfs_write};
use stem::{info, warn};

#[stem::main]
fn main(_arg: usize) -> ! {
    info!("ipc_pipe_demo: starting up");

    // ── 1. Create the pipe ────────────────────────────────────────────────
    //
    // pipe() fills a [u32; 2] with [read_fd, write_fd].
    let mut pipefds = [0u32; 2];
    match pipe(&mut pipefds) {
        Ok(()) => {}
        Err(e) => {
            warn!("ipc_pipe_demo: pipe() failed: {:?}", e);
            stem::syscall::exit(1);
        }
    }
    let (pr, pw) = (pipefds[0], pipefds[1]);
    info!("ipc_pipe_demo: created pipe read_fd={} write_fd={}", pr, pw);

    // ── 2. Write data to the write-end ────────────────────────────────────
    //
    // In a real scenario this would be the child writing to its stdout.
    // Here we do it in-process to keep the demo self-contained.
    let msg = b"Hello from the write-end!\n";
    match vfs_write(pw, msg) {
        Ok(n) => info!("ipc_pipe_demo: wrote {} bytes", n),
        Err(e) => {
            warn!("ipc_pipe_demo: write failed: {:?}", e);
            stem::syscall::exit(1);
        }
    }

    // ── 3. Close the write-end ────────────────────────────────────────────
    //
    // Once closed, the kernel will signal EOF to any reader when the buffer
    // is drained.  This is the standard "close your end" convention.
    vfs_close(pw).expect("close write-end");
    info!("ipc_pipe_demo: closed write-end");

    // ── 4. Read from the read-end until EOF ───────────────────────────────
    let mut buf = [0u8; 256];
    let mut total = 0usize;
    loop {
        match vfs_read(pr, &mut buf) {
            Ok(0) => {
                info!("ipc_pipe_demo: EOF after {} bytes", total);
                break;
            }
            Ok(n) => {
                total += n;
                let s = core::str::from_utf8(&buf[..n]).unwrap_or("<binary>");
                info!("ipc_pipe_demo: read {} bytes: {:?}", n, s);
            }
            Err(e) => {
                warn!("ipc_pipe_demo: read error: {:?}", e);
                break;
            }
        }
    }

    vfs_close(pr).expect("close read-end");
    info!("ipc_pipe_demo: PASS — pipe round-trip complete");
    stem::syscall::exit(0);
}
