//! User stack allocation and fault handling.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::memory::MapError;
use crate::{BootRuntime, BootTasking, MapKind, MapPerms, memory};

use super::SCHEDULER;
use super::types::{Scheduler, StackFaultResult};

const DEFAULT_USER_STACK_PAGES: usize = 16;
const MAX_USER_STACK_PAGES: usize = 256;

static NEXT_USER_STACK: AtomicU64 = AtomicU64::new(0x7FFF_0000_0000);

pub fn alloc_user_stack<R: BootRuntime>(pages: usize) -> Option<usize> {
    let rt = crate::runtime::<R>();
    let page_size = rt.page_size() as u64;

    let requested_pages = if pages == 0 {
        DEFAULT_USER_STACK_PAGES
    } else {
        pages
    };
    let clamped_pages = core::cmp::min(requested_pages, MAX_USER_STACK_PAGES);
    let total_size = (clamped_pages as u64).saturating_mul(page_size);

    let base = NEXT_USER_STACK.fetch_add(total_size, Ordering::SeqCst);
    let top = base + total_size;

    let aspace = rt.tasking().active_address_space();
    let perms = MapPerms {
        user: true,
        read: true,
        write: true,
        exec: false,
        kind: MapKind::Normal,
    };
    let hook = crate::GlobalAllocHook;

    let mut virt = base;
    for _ in 0..clamped_pages {
        let phys = memory::alloc_frame()?;
        let hhdm_virt = phys + rt.phys_to_virt_offset();
        unsafe {
            core::ptr::write_bytes(hhdm_virt as *mut u8, 0, page_size as usize);
        }
        rt.tasking()
            .map_page(aspace, virt, phys, perms, MapKind::Normal, &hook)
            .ok()?;
        virt += page_size;
    }

    Some(top as usize)
}

/// Map a user page in the current address space
pub unsafe fn map_user_page<R: BootRuntime>(virt: u64, phys: u64) -> Result<(), MapError> {
    use crate::FrameAllocatorHook;

    struct MapHook;
    impl FrameAllocatorHook for MapHook {
        fn alloc_frame(&self) -> Option<u64> {
            crate::memory::alloc_frame()
        }
    }

    let rt = crate::runtime::<R>();
    let aspace = rt.tasking().active_address_space();
    let perms = MapPerms {
        user: true,
        read: true,
        write: true,
        exec: false,
        kind: MapKind::Normal,
    };
    let hook = MapHook;

    rt.tasking()
        .map_page(aspace, virt, phys, perms, MapKind::Normal, &hook)
        .map_err(|()| MapError::OutOfMemory)?;
    rt.tasking().tlb_flush_page(virt);

    Ok(())
}

/// Map a user page with explicit permissions in the current address space.
pub unsafe fn map_user_page_perms<R: BootRuntime>(
    virt: u64,
    phys: u64,
    perms: MapPerms,
) -> Result<(), MapError> {
    use crate::FrameAllocatorHook;

    struct MapHook;
    impl FrameAllocatorHook for MapHook {
        fn alloc_frame(&self) -> Option<u64> {
            crate::memory::alloc_frame()
        }
    }

    let rt = crate::runtime::<R>();
    let aspace = rt.tasking().active_address_space();
    let hook = MapHook;

    rt.tasking()
        .map_page(aspace, virt, phys, perms, perms.kind, &hook)
        .map_err(|()| MapError::OutOfMemory)?;
    rt.tasking().tlb_flush_page(virt);
    Ok(())
}

/// Unmap a user page in the current address space
pub unsafe fn unmap_user_page<R: BootRuntime>(virt: u64) -> Result<(), MapError> {
    let rt = crate::runtime::<R>();
    let aspace = rt.tasking().active_address_space();

    rt.tasking()
        .unmap_page(aspace, virt)
        .map_err(|()| MapError::NotMapped)?;
    rt.tasking().tlb_flush_page(virt);
    Ok(())
}

/// Protect a user page with explicit permissions in the current address space.
pub unsafe fn protect_user_page<R: BootRuntime>(
    virt: u64,
    perms: MapPerms,
) -> Result<(), MapError> {
    let rt = crate::runtime::<R>();
    let aspace = rt.tasking().active_address_space();

    rt.tasking()
        .protect_page(aspace, virt, perms)
        .map_err(|()| MapError::PageTableFault)?;
    rt.tasking().tlb_flush_page(virt);
    Ok(())
}

pub unsafe fn handle_stack_fault<R: BootRuntime>(addr: u64) -> StackFaultResult {
    let rt = crate::runtime::<R>();
    let page_size = rt.page_size() as u64;

    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let ptr = match *lock {
        Some(ptr) => ptr,
        None => {
            rt.irq_restore(_irq);
            return StackFaultResult::NotStack;
        }
    };
    let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
    let cpu = super::current_cpu_index::<R>();
    let current_id = match sched.state.per_cpu.get(cpu).and_then(|pc| pc.current) {
        Some(id) => id,
        None => {
            rt.irq_restore(_irq);
            return StackFaultResult::NotStack;
        }
    };
    let mut task = match crate::task::registry::get_task_mut::<R>(current_id) {
        Some(t) => t,
        None => {
            rt.irq_restore(_irq);
            return StackFaultResult::NotStack;
        }
    };

    let info = match task.stack_info {
        Some(info) => info,
        None => {
            rt.irq_restore(_irq);
            return StackFaultResult::NotStack;
        }
    };

    let guard_start = info.guard_start as u64;
    let guard_end = info.guard_end as u64;
    let reserve_start = info.reserve_start as u64;
    let reserve_end = info.reserve_end as u64;
    let committed_start = info.committed_start as u64;

    if addr >= guard_start && addr < guard_end {
        rt.irq_restore(_irq);
        return StackFaultResult::Overflow;
    }

    if addr < reserve_start || addr >= reserve_end || addr >= committed_start {
        rt.irq_restore(_irq);
        return StackFaultResult::NotStack;
    }

    let fault_page = addr & !(page_size - 1);
    let grow_chunk = core::cmp::max(info.grow_chunk_bytes as u64, page_size);
    let mut new_commit_start = committed_start.saturating_sub(grow_chunk);
    new_commit_start &= !(page_size - 1);
    if new_commit_start > fault_page {
        new_commit_start = fault_page;
    }
    if new_commit_start < reserve_start {
        new_commit_start = reserve_start;
    }

    if new_commit_start == committed_start {
        rt.irq_restore(_irq);
        return StackFaultResult::NotStack;
    }

    let hhdm = crate::boot_info::get().map(|i| i.hhdm_offset).unwrap_or(0);
    let perms = MapPerms {
        user: true,
        read: true,
        write: true,
        exec: false,
        kind: MapKind::Normal,
    };

    let mut virt = new_commit_start;
    while virt < committed_start {
        let phys = match crate::memory::alloc_frame() {
            Some(p) => p,
            None => {
                rt.irq_restore(_irq);
                return StackFaultResult::NotStack;
            }
        };
        let hhdm_virt = phys + hhdm;
        unsafe {
            core::ptr::write_bytes(hhdm_virt as *mut u8, 0, page_size as usize);
        }
        if unsafe { crate::memory::map_user_page_with_perms(virt, phys, perms) }.is_err() {
            rt.irq_restore(_irq);
            return StackFaultResult::NotStack;
        }
        virt += page_size;
    }

    task.stack_info = Some(abi::types::StackInfo {
        committed_start: new_commit_start as usize,
        ..info
    });

    rt.irq_restore(_irq);
    StackFaultResult::Grew
}
