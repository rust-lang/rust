//! Global Allocator for ThingOS.
//!
//! Implements the `System` allocator using `dlmalloc` as the heap manager,
//! backed by anonymous memory pages obtained from the kernel via `SYS_VM_MAP`
//! and released via `SYS_VM_UNMAP`.
//!
//! When a ThingOS userspace binary also links `stem` (which registers its own
//! `#[global_allocator]`), that registration takes precedence and these methods
//! are never called.  However, any code that calls `System` directly — or any
//! binary that does *not* link stem — will use this correct implementation
//! instead of silently getting null pointers.

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;
use crate::sync::atomic::Ordering::{Acquire, Release};
use crate::sync::atomic::{Atomic, AtomicI32};
use crate::sys::pal::raw_syscall6;

// ── Syscall numbers (abi/src/numbers.rs) ─────────────────────────────────────
const SYS_VM_MAP: u32 = 0x2001;
const SYS_VM_UNMAP: u32 = 0x2002;

// ── Protection flags (mirrors abi::vm::VmProt bits) ──────────────────────────
const PROT_READ: u32 = 1 << 0;
const PROT_WRITE: u32 = 1 << 1;
const PROT_USER: u32 = 1 << 3;

// ── Map flags (mirrors abi::vm::VmMapFlags bits) ─────────────────────────────
const MAP_PRIVATE: u32 = 1 << 2;

// ── Local mirror of abi::vm::VmBacking (repr(C) enum) ────────────────────────
//
// The repr(C) tagged-union layout on a 64-bit target is:
//   offset  0: discriminant (i32, 4 bytes) — 0 = Anonymous
//   offset  4: padding (4 bytes, to align the data union to 8)
//   offset  8: union data (16 bytes — sized by the File variant: u32 + u64)
//
// For Anonymous { zeroed: bool }, `zeroed` lives at offset 8 (first byte of
// the union), and the remaining 15 bytes are unused padding.
#[repr(C)]
struct VmBackingAnon {
    discriminant: i32, // 0 = Anonymous
    _pad0: [u8; 4],
    zeroed: u8, // 0 = false, 1 = true
    _pad1: [u8; 15],
}

// ── Local mirrors of abi::vm request/response types ──────────────────────────
#[repr(C)]
struct VmMapReq {
    addr_hint: usize,
    len: usize,
    prot: u32,
    flags: u32,
    backing: VmBackingAnon,
}

#[repr(C)]
struct VmMapResp {
    addr: usize,
    len: usize,
}

#[repr(C)]
struct VmUnmapReq {
    addr: usize,
    len: usize,
}

#[repr(C)]
struct VmUnmapResp {
    unmapped_len: usize,
}

// ── ThingOS dlmalloc system allocator ────────────────────────────────────────

struct ThingOs;

const PAGE_SIZE: usize = 0x1000;

#[inline]
fn page_align(n: usize) -> usize {
    (n + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

unsafe impl dlmalloc::Allocator for ThingOs {
    fn alloc(&self, size: usize) -> (*mut u8, usize, u32) {
        let len = page_align(size);
        if len == 0 {
            return (ptr::null_mut(), 0, 0);
        }
        let req = VmMapReq {
            addr_hint: 0,
            len,
            prot: PROT_READ | PROT_WRITE | PROT_USER,
            flags: MAP_PRIVATE,
            backing: VmBackingAnon {
                discriminant: 0, // Anonymous
                _pad0: [0; 4],
                zeroed: 1, // request zeroed pages
                _pad1: [0; 15],
            },
        };
        let mut resp = VmMapResp { addr: 0, len: 0 };
        let ret = unsafe {
            raw_syscall6(
                SYS_VM_MAP,
                &req as *const VmMapReq as usize,
                &mut resp as *mut VmMapResp as usize,
                0,
                0,
                0,
                0,
            )
        };
        if ret < 0 || resp.addr == 0 {
            (ptr::null_mut(), 0, 0)
        } else {
            (ptr::with_exposed_provenance_mut(resp.addr), resp.len, 0)
        }
    }

    fn remap(&self, _ptr: *mut u8, _oldsize: usize, _newsize: usize, _can_move: bool) -> *mut u8 {
        // ThingOS has no mremap equivalent; let dlmalloc fall back to alloc+copy+free.
        ptr::null_mut()
    }

    fn free_part(&self, ptr: *mut u8, oldsize: usize, newsize: usize) -> bool {
        // Unmap the tail pages that are no longer needed.
        let keep = page_align(newsize);
        let old_pages = page_align(oldsize);
        if keep >= old_pages {
            return true;
        }
        let free_start = ptr as usize + keep;
        let free_len = old_pages - keep;
        let req = VmUnmapReq { addr: free_start, len: free_len };
        let mut resp = VmUnmapResp { unmapped_len: 0 };
        let ret = unsafe {
            raw_syscall6(
                SYS_VM_UNMAP,
                &req as *const VmUnmapReq as usize,
                &mut resp as *mut VmUnmapResp as usize,
                0,
                0,
                0,
                0,
            )
        };
        ret >= 0
    }

    fn free(&self, ptr: *mut u8, size: usize) -> bool {
        let len = page_align(size);
        if len == 0 {
            return true;
        }
        let req = VmUnmapReq { addr: ptr as usize, len };
        let mut resp = VmUnmapResp { unmapped_len: 0 };
        let ret = unsafe {
            raw_syscall6(
                SYS_VM_UNMAP,
                &req as *const VmUnmapReq as usize,
                &mut resp as *mut VmUnmapResp as usize,
                0,
                0,
                0,
                0,
            )
        };
        ret >= 0
    }

    fn can_release_part(&self, _flags: u32) -> bool {
        true
    }

    fn allocates_zeros(&self) -> bool {
        true // we request zeroed pages from the kernel
    }

    fn page_size(&self) -> usize {
        PAGE_SIZE
    }
}

// ── Locked dlmalloc instance ──────────────────────────────────────────────────
//
// Use a simple spinlock so the allocator is usable even before the thread
// subsystem (mutexes) is fully initialised.

#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys5alloc7thingos8DLMALLOCE")]
static DLMALLOC: AllocLock =
    AllocLock { lock: AtomicI32::new(0), inner: dlmalloc::Dlmalloc::new_with_allocator(ThingOs) };

struct AllocLock {
    lock: Atomic<i32>,
    inner: dlmalloc::Dlmalloc<ThingOs>,
}

// SAFETY: The `AllocLock` is only accessed through the `lock()`/`unlock()` pair
// below, which provides exclusive access.
unsafe impl Sync for AllocLock {}

struct AllocGuard<'a>(&'a AllocLock);

impl AllocLock {
    fn lock(&self) -> AllocGuard<'_> {
        loop {
            if self.lock.swap(1, Acquire) == 0 {
                return AllocGuard(self);
            }
            // Spin — the allocator critical section is very short.
            core::hint::spin_loop();
        }
    }
}

impl Drop for AllocGuard<'_> {
    fn drop(&mut self) {
        self.0.lock.store(0, Release);
    }
}

impl AllocGuard<'_> {
    fn dlmalloc(&mut self) -> &mut dlmalloc::Dlmalloc<ThingOs> {
        // SAFETY: we hold the lock, so we have exclusive access.
        unsafe { &mut *(&self.0.inner as *const _ as *mut _) }
    }
}

// ── System GlobalAlloc impl ───────────────────────────────────────────────────

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut guard = DLMALLOC.lock();
        // SAFETY: dlmalloc preconditions match GlobalAlloc::alloc preconditions.
        unsafe { guard.dlmalloc().malloc(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let mut guard = DLMALLOC.lock();
        // SAFETY: dlmalloc preconditions match GlobalAlloc::alloc_zeroed preconditions.
        unsafe { guard.dlmalloc().calloc(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let mut guard = DLMALLOC.lock();
        // SAFETY: dlmalloc preconditions match GlobalAlloc::dealloc preconditions.
        unsafe { guard.dlmalloc().free(ptr, layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let mut guard = DLMALLOC.lock();
        // SAFETY: dlmalloc preconditions match GlobalAlloc::realloc preconditions.
        unsafe { guard.dlmalloc().realloc(ptr, layout.size(), layout.align(), new_size) }
    }
}
