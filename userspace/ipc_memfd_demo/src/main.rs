//! ipc_memfd_demo — IPC Cookbook Recipe 5: memfd-backed shared buffer exchange.
//!
//! This program demonstrates how to use memory-file descriptors (`memfd`) to
//! share large buffers between producers and consumers without copying the
//! data.
//!
//! # What this shows
//!
//! 1. **Create a memfd** — `memfd_create` allocates a named anonymous memory
//!    region.
//! 2. **Map the memfd** — `vm_map` maps the region into the process's address
//!    space so it can be accessed as a regular slice.
//! 3. **Write to the mapping** — the producer fills the buffer.
//! 4. **Unmap and re-map** — verifies that data survives an unmap/remap cycle
//!    (simulating what a compositor would do after receiving the fd).
//! 5. **Pass the fd** — in a real multi-process scenario the fd would be
//!    transferred to a consumer process via `channel_send_msg`.  See
//!    `docs/concepts/ipc_cookbook.md` Recipe 5 for the full multi-process
//!    version.
//!
//! # Running
//!
//! ```text
//! # In the Thing-OS shell:
//! $ ipc_memfd_demo
//! ```
//!
//! # See Also
//!
//! `docs/concepts/ipc_cookbook.md` Recipe 5 — memfd-backed shared buffer.
//! `docs/concepts/memfd.md` — memfd deep-dive.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
use stem::syscall::{memfd_create, vfs_close, vm_map, vm_unmap};
use stem::{info, warn};

/// Demonstration buffer size (4 KiB).
const BUF_SIZE: usize = 4096;

#[stem::main]
fn main(_arg: usize) -> ! {
    info!("ipc_memfd_demo: starting up");

    // ── 1. Create a memfd ─────────────────────────────────────────────────
    //
    // memfd_create returns an fd that backs an anonymous memory region.
    // The name is only used for diagnostics; it does not appear in any
    // directory.
    let fd = match memfd_create("demo-buffer", BUF_SIZE) {
        Ok(fd) => {
            info!("ipc_memfd_demo: memfd created fd={}", fd);
            fd
        }
        Err(e) => {
            warn!("ipc_memfd_demo: memfd_create failed: {:?}", e);
            stem::syscall::exit(1);
        }
    };

    // ── 2. Map the memfd for read/write ───────────────────────────────────
    let req = VmMapReq {
        addr_hint: 0,
        len: BUF_SIZE,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::SHARED,
        backing: VmBacking::File { fd, offset: 0 },
    };
    let mapped_rw = match vm_map(&req) {
        Ok(resp) => {
            info!("ipc_memfd_demo: mapped RW at 0x{:x}", resp.addr);
            resp
        }
        Err(e) => {
            warn!("ipc_memfd_demo: vm_map (RW) failed: {:?}", e);
            let _ = vfs_close(fd);
            stem::syscall::exit(1);
        }
    };

    // ── 3. Write a pattern to the buffer ─────────────────────────────────
    //
    // SAFETY: `mapped_rw.addr` is valid for `BUF_SIZE` bytes because the
    // kernel just mapped it for us.
    let pattern: u8 = 0xAB;
    unsafe {
        let slice = core::slice::from_raw_parts_mut(mapped_rw.addr as *mut u8, BUF_SIZE);
        for b in slice.iter_mut() {
            *b = pattern;
        }
    }
    info!(
        "ipc_memfd_demo: filled buffer with pattern 0x{:02X}",
        pattern
    );

    // ── 4. Unmap the RW mapping ───────────────────────────────────────────
    //
    // In a real driver the sender would unmap after passing the fd to the
    // receiver.  The backing memory is retained as long as any fd or mapping
    // refers to it.
    if let Err(e) = vm_unmap(mapped_rw.addr, BUF_SIZE) {
        warn!("ipc_memfd_demo: vm_unmap (RW) failed: {:?}", e);
    } else {
        info!("ipc_memfd_demo: unmapped RW mapping");
    }

    // ── 5. Re-map read-only and verify the pattern ────────────────────────
    //
    // Simulates the consumer (e.g. compositor) mapping the received fd.
    let req_ro = VmMapReq {
        addr_hint: 0,
        len: BUF_SIZE,
        prot: VmProt::READ | VmProt::USER,
        flags: VmMapFlags::SHARED,
        backing: VmBacking::File { fd, offset: 0 },
    };
    let mapped_ro = match vm_map(&req_ro) {
        Ok(resp) => {
            info!("ipc_memfd_demo: re-mapped RO at 0x{:x}", resp.addr);
            resp
        }
        Err(e) => {
            warn!("ipc_memfd_demo: vm_map (RO) failed: {:?}", e);
            let _ = vfs_close(fd);
            stem::syscall::exit(1);
        }
    };

    // Verify every byte matches the written pattern.
    let ok = unsafe {
        let slice = core::slice::from_raw_parts(mapped_ro.addr as *const u8, BUF_SIZE);
        slice.iter().all(|&b| b == pattern)
    };

    if ok {
        info!(
            "ipc_memfd_demo: PASS — all {} bytes match 0x{:02X}",
            BUF_SIZE, pattern
        );
    } else {
        warn!("ipc_memfd_demo: FAIL — buffer mismatch!");
    }

    // ── 6. Clean up ───────────────────────────────────────────────────────
    if let Err(e) = vm_unmap(mapped_ro.addr, BUF_SIZE) {
        warn!("ipc_memfd_demo: vm_unmap (RO) failed: {:?}", e);
    }
    vfs_close(fd).expect("close memfd");

    info!("ipc_memfd_demo: done");
    stem::syscall::exit(0);
}
