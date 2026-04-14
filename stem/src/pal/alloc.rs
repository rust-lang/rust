//! Memory allocator platform abstraction.
//!
//! Provides heap growth primitives that interface with the virtual memory
//! subsystem via syscalls.

use abi::errors::Errno;
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
use spin::Mutex;

use crate::utils::align_up;
use crate::vm::vm_map;

const PAGE_SIZE: usize = 4096;
const HEAP_BASE: usize = 0x2000_0000;
const HEAP_GROW_MIN: usize = 256 * 1024;

/// Heap state tracking.
struct HeapState {
    base: usize,
    size: usize,
    initialized: bool,
}

static HEAP_STATE: Mutex<HeapState> = Mutex::new(HeapState {
    base: HEAP_BASE,
    size: 0,
    initialized: false,
});

/// Grow the heap by at least `min_bytes`.
///
/// This is the platform primitive for heap expansion. It maps additional
/// anonymous memory and extends the allocator's managed region.
///
/// Returns an error if the mapping fails or if the kernel cannot satisfy
/// the request at the expected address.
pub fn grow_heap(min_bytes: usize) -> Result<(), Errno> {
    let mut state = HEAP_STATE.lock();
    let grow_bytes = align_up(core::cmp::max(min_bytes, HEAP_GROW_MIN), PAGE_SIZE);
    let map_addr = state.base + state.size;

    let req = VmMapReq {
        addr_hint: map_addr,
        len: grow_bytes,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::FIXED | VmMapFlags::PRIVATE,
        backing: VmBacking::Anonymous { zeroed: true },
    };
    let resp = vm_map(&req)?;
    if resp.addr != map_addr {
        return Err(Errno::ENOMEM);
    }

    if state.initialized {
        #[cfg(feature = "global-alloc")]
        crate::allocator::extend_heap(grow_bytes);
    } else {
        #[cfg(feature = "global-alloc")]
        crate::allocator::init_heap(map_addr, grow_bytes);
        state.initialized = true;
    }
    state.size += grow_bytes;
    Ok(())
}
