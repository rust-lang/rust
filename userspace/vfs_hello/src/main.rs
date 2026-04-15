//! vfs_hello — "Hello from VFS" milestone program.
//!
//! This program demonstrates Act III of the thingos de-graphing migration:
//! the kernel VFS is alive when this program can:
//!
//! 1. `open("/dev/console")` via `SYS_FS_OPEN`
//! 2. `write(fd, "Hello from VFS!\n")` via `SYS_FS_WRITE`
//! 3. `close(fd)` via `SYS_FS_CLOSE`
//!
//! A second pass verifies `/dev/null` and `/dev/zero` are also reachable.
//!
//! # North star
//! You can debug thingos with `cat` and `ls`.
//! This program is the first step towards that world.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

#[stem::main]
fn main() -> ! {
    use abi::syscall::vfs_flags::{O_RDONLY, O_WRONLY};
    use stem::info;
    use stem::syscall::{vfs_close, vfs_open, vfs_read, vfs_write};

    // ── /dev/console write ────────────────────────────────────────────────
    match vfs_open("/dev/console", O_WRONLY) {
        Ok(fd) => {
            let msg = b"Hello from VFS!\n";
            match vfs_write(fd, msg) {
                Ok(n) => {
                    // Also log via the legacy stem logger so it shows in the
                    // kernel serial log regardless of which console driver is
                    // active.
                    info!("vfs_hello: wrote {} bytes to /dev/console (fd {})", n, fd);
                }
                Err(e) => {
                    info!("vfs_hello: write to /dev/console failed: {:?}", e);
                }
            }
            let _ = vfs_close(fd);
        }
        Err(e) => {
            info!("vfs_hello: open /dev/console failed: {:?}", e);
        }
    }

    // ── /dev/null smoke test ──────────────────────────────────────────────
    match vfs_open("/dev/null", O_WRONLY) {
        Ok(fd) => {
            let _ = vfs_write(fd, b"discard me\n");
            let _ = vfs_close(fd);
            info!("vfs_hello: /dev/null OK");
        }
        Err(e) => {
            info!("vfs_hello: open /dev/null failed: {:?}", e);
        }
    }

    // ── /dev/zero smoke test ──────────────────────────────────────────────
    match vfs_open("/dev/zero", O_RDONLY) {
        Ok(fd) => {
            let mut buf = [0xFFu8; 8];
            match vfs_read(fd, &mut buf) {
                Ok(n) if n == 8 && buf.iter().all(|&b| b == 0) => {
                    info!("vfs_hello: /dev/zero OK (read {} zero bytes)", n);
                }
                Ok(n) => {
                    info!(
                        "vfs_hello: /dev/zero unexpected: n={} buf={:?}",
                        n,
                        &buf[..n]
                    );
                }
                Err(e) => {
                    info!("vfs_hello: read /dev/zero failed: {:?}", e);
                }
            }
            let _ = vfs_close(fd);
        }
        Err(e) => {
            info!("vfs_hello: open /dev/zero failed: {:?}", e);
        }
    }

    info!("vfs_hello: done");
    stem::syscall::exit(0);
}
