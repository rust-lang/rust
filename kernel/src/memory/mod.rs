use crate::{BootTasking, MapPerms};
pub mod arena;
pub mod boot_frame_alloc;
pub mod boot_heap;
pub mod frame_alloc;
pub mod global_alloc;
pub mod handle;
pub mod kheap;
pub mod layout;
pub mod map;
pub mod mappings;
pub mod paging;

use crate::kinfo;
pub use frame_alloc::FRAME_ALLOCATOR;
use spin::Mutex;

/// Next user VA for mappings (starts at 0x1000_0000, grows up)
static NEXT_MAP_VA: Mutex<u64> = Mutex::new(0x1000_0000);

/// Allocate a user VA range. Simple bump allocator for v0.
pub fn alloc_user_va(size: usize) -> u64 {
    let mut next = NEXT_MAP_VA.lock();
    let va = *next;
    // Align to page boundary and bump
    *next = (*next + size as u64 + 4095) & !4095;
    va
}

pub fn init<R: crate::BootRuntime>(rt: &R) {
    let map = rt.phys_memory_map();
    let _modules = rt.modules();
    let offset = rt.phys_to_virt_offset();

    crate::kdebug!("Memory map has {} entries", map.len());
    for (i, range) in map.iter().enumerate() {
        crate::ktrace!(
            "  [{}] 0x{:x} - 0x{:x} ({:?})",
            i,
            range.start,
            range.end,
            range.kind
        );
    }
    crate::kdebug!("HHDM Offset: 0x{:x}", offset);

    // 1. Setup early frame allocator
    let bitmap = boot_frame_alloc::init(map, offset);
    let alloc = frame_alloc::FrameAllocator::new_from_boot(map, _modules, bitmap, offset);

    crate::kdebug!(
        "Frame allocator initialized with {} free frames",
        alloc.free_count()
    );

    unsafe { FRAME_ALLOCATOR.init(alloc) };

    rt.tasking().init(offset);
}

pub fn alloc_frame() -> Option<u64> {
    FRAME_ALLOCATOR.with_lock(|a| a.alloc().map(|f| f.0))
}

/// Allocate `count` physically contiguous 4K frames.
/// Returns the physical base address if successful.
pub fn alloc_contiguous_frames(count: usize) -> Option<u64> {
    FRAME_ALLOCATOR.with_lock(|a| a.alloc_contiguous(count))
}

/// Free `count` physically contiguous 4K frames starting at `phys`.
pub fn free_contiguous_frames(phys: u64, count: usize) {
    FRAME_ALLOCATOR.with_lock(|a| a.mark_free_range(phys, phys + (count as u64 * 4096)));
}

/// Global hook for mapping user pages. Set by scheduler init.
static mut MAP_USER_PAGE_HOOK: Option<unsafe fn(u64, u64) -> Result<(), MapError>> = None;
static mut MAP_USER_PAGE_PERMS_HOOK: Option<unsafe fn(u64, u64, MapPerms) -> Result<(), MapError>> =
    None;
static mut UNMAP_USER_PAGE_HOOK: Option<unsafe fn(u64) -> Result<(), MapError>> = None;
static mut PROTECT_USER_PAGE_HOOK: Option<unsafe fn(u64, MapPerms) -> Result<(), MapError>> = None;

/// Error from user page mapping operations.
///
/// Provides typed error information that can be mapped to appropriate errno values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapError {
    /// Failed to allocate page table frame
    OutOfMemory,
    /// Page table walk failed (corrupt or invalid)
    PageTableFault,
    /// Page is already mapped
    AlreadyMapped,
    /// Page is not mapped (for unmap)
    NotMapped,
    /// Generic permission/access error
    AccessDenied,
}

impl MapError {
    /// Convert to the appropriate errno for syscall return
    pub fn to_errno(self) -> abi::errors::Errno {
        match self {
            MapError::OutOfMemory => abi::errors::Errno::ENOMEM,
            MapError::PageTableFault => abi::errors::Errno::EFAULT,
            MapError::AlreadyMapped => abi::errors::Errno::EEXIST,
            MapError::NotMapped => abi::errors::Errno::EINVAL,
            MapError::AccessDenied => abi::errors::Errno::EACCES,
        }
    }
}

/// Initialize the user page mapping hook
pub unsafe fn set_map_user_page_hook(hook: unsafe fn(u64, u64) -> Result<(), MapError>) {
    unsafe { MAP_USER_PAGE_HOOK = Some(hook) };
}

/// Initialize the user page mapping hook with custom permissions.
pub unsafe fn set_map_user_page_perms_hook(
    hook: unsafe fn(u64, u64, MapPerms) -> Result<(), MapError>,
) {
    unsafe { MAP_USER_PAGE_PERMS_HOOK = Some(hook) };
}

/// Initialize the user page unmapping hook.
pub unsafe fn set_unmap_user_page_hook(hook: unsafe fn(u64) -> Result<(), MapError>) {
    unsafe { UNMAP_USER_PAGE_HOOK = Some(hook) };
}

/// Initialize the user page protection hook.
pub unsafe fn set_protect_user_page_hook(hook: unsafe fn(u64, MapPerms) -> Result<(), MapError>) {
    unsafe { PROTECT_USER_PAGE_HOOK = Some(hook) };
}

/// Map a physical page into the current process's userspace at the given virtual address.
/// This uses the global hook set during scheduler initialization.
pub unsafe fn map_user_page(virt: u64, phys: u64) -> Result<(), abi::errors::Errno> {
    if let Some(hook) = unsafe { MAP_USER_PAGE_HOOK } {
        unsafe { hook(virt, phys) }.map_err(|e| e.to_errno())
    } else {
        kinfo!(
            "WARN: map_user_page called before hook installed (virt=0x{:x})",
            virt
        );
        Err(abi::errors::Errno::EIO)
    }
}

/// Unmap a page from the current process's userspace.
pub unsafe fn unmap_user_page(virt: u64) -> Result<(), abi::errors::Errno> {
    if let Some(hook) = unsafe { UNMAP_USER_PAGE_HOOK } {
        unsafe { hook(virt) }.map_err(|e| e.to_errno())
    } else {
        kinfo!(
            "WARN: unmap_user_page called before hook installed (virt=0x{:x})",
            virt
        );
        Err(abi::errors::Errno::EIO)
    }
}

/// Global hook for translating user addresses.
static mut TRANSLATE_USER_PAGE_HOOK: Option<unsafe fn(u64) -> Option<u64>> = None;

/// Initialize the user page translation hook.
pub unsafe fn set_translate_user_page_hook(hook: unsafe fn(u64) -> Option<u64>) {
    unsafe { TRANSLATE_USER_PAGE_HOOK = Some(hook) };
}

/// Translate a user virtual address to a physical address.
pub fn translate_user_page(virt: u64) -> Option<u64> {
    if let Some(hook) = unsafe { TRANSLATE_USER_PAGE_HOOK } {
        unsafe { hook(virt) }
    } else {
        None
    }
}

/// Map a physical page into the current process's userspace with explicit permissions.
pub unsafe fn map_user_page_with_perms(
    virt: u64,
    phys: u64,
    perms: MapPerms,
) -> Result<(), abi::errors::Errno> {
    if let Some(hook) = unsafe { MAP_USER_PAGE_PERMS_HOOK } {
        unsafe { hook(virt, phys, perms) }.map_err(|e| e.to_errno())
    } else {
        kinfo!(
            "WARN: map_user_page_with_perms called before hook installed (virt=0x{:x})",
            virt
        );
        Err(abi::errors::Errno::EIO)
    }
}

/// Change protection for a user page.
pub unsafe fn protect_user_page(virt: u64, perms: MapPerms) -> Result<(), abi::errors::Errno> {
    if let Some(hook) = unsafe { PROTECT_USER_PAGE_HOOK } {
        unsafe { hook(virt, perms) }.map_err(|e| e.to_errno())
    } else {
        kinfo!(
            "WARN: protect_user_page called before hook installed (virt=0x{:x})",
            virt
        );
        Err(abi::errors::Errno::EIO)
    }
}
