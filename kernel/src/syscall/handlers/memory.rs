//! Memory and stack allocation syscalls

use abi::errors::{Errno, SysResult};
use abi::vm::{
    VmBacking, VmBackingKind, VmMapReq, VmMapResp, VmProt, VmRegionInfo, VmUnmapReq, VmUnmapResp,
};
use core::sync::atomic::{AtomicU64, Ordering};

/// Base address for anonymous VM mappings (stacks, etc).
/// Uses high user VA space to avoid collision with bytespace mappings at 0x1000_0000.
const USER_VM_BASE: u64 = 0x4000_0000_0000; // 64TB mark
static NEXT_USER_MAP: AtomicU64 = AtomicU64::new(USER_VM_BASE);

pub fn sys_alloc_stack(pages: usize) -> SysResult<usize> {
    let top = unsafe { crate::sched::alloc_user_stack_current(pages) }.ok_or(Errno::ENOMEM)?;
    Ok(top)
}

pub fn sys_vm_map(req_ptr: usize, resp_ptr: usize) -> SysResult<usize> {
    use crate::syscall::validate::{copyin, copyout, validate_user_range};

    let req_size = core::mem::size_of::<VmMapReq>();
    let resp_size = core::mem::size_of::<VmMapResp>();
    validate_user_range(req_ptr, req_size, false)?;
    validate_user_range(resp_ptr, resp_size, true)?;

    let mut req: VmMapReq = unsafe { core::mem::zeroed() };
    let req_slice =
        unsafe { core::slice::from_raw_parts_mut(&mut req as *mut _ as *mut u8, req_size) };
    unsafe {
        copyin(req_slice, req_ptr)?;
    }

    let page_size = 4096usize;
    let len = align_up(req.len, page_size);
    if len == 0 {
        return Err(Errno::EINVAL);
    }

    let fixed = req.flags.contains(abi::vm::VmMapFlags::FIXED);
    let guard = req.flags.contains(abi::vm::VmMapFlags::GUARD);

    // Legacy compatibility: callers that don't set PRIVATE/SHARED get a
    // sensible default based on backing type.
    let map_private = req.flags.contains(abi::vm::VmMapFlags::PRIVATE);
    let map_shared = req.flags.contains(abi::vm::VmMapFlags::SHARED);
    if map_private && map_shared {
        return Err(Errno::EINVAL);
    }

    let effective_shared = if map_shared {
        true
    } else if map_private {
        false
    } else {
        matches!(req.backing, VmBacking::File { .. })
    };

    let mut addr = req.addr_hint;
    if fixed {
        if addr == 0 || addr % page_size != 0 {
            return Err(Errno::EINVAL);
        }
    } else {
        if addr == 0 {
            addr = NEXT_USER_MAP.fetch_add(len as u64, Ordering::SeqCst) as usize;
        }
        addr = align_up(addr, page_size);
    }

    if addr % page_size != 0 {
        return Err(Errno::EINVAL);
    }

    if !guard {
        let perms = map_perms_from_prot(req.prot);
        let hhdm = crate::boot_info::get().map(|i| i.hhdm_offset).unwrap_or(0);
        let mut virt = addr as u64;
        let end = virt + len as u64;
        // Resolve file backing metadata once to avoid re-locking fd_table on
        // every page and to enforce access checks up front.
        let file_backing = match req.backing {
            VmBacking::File { fd, offset } => {
                let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
                let (node, status_flags) = {
                    let lock = pinfo_arc.lock();
                    let file = lock.fd_table.get(fd)?;
                    (file.node.clone(), *file.status_flags.lock())
                };

                if req.prot.contains(VmProt::READ) && !status_flags.is_readable() {
                    return Err(Errno::EACCES);
                }
                if req.prot.contains(VmProt::WRITE) && !status_flags.is_writable() {
                    return Err(Errno::EACCES);
                }

                let shared_region = if effective_shared {
                    let (phys_base, total_size) =
                        node.phys_region().map_err(|_| Errno::EOPNOTSUPP)?;
                    if offset as usize >= total_size || total_size - (offset as usize) < len {
                        return Err(Errno::EINVAL);
                    }
                    Some((phys_base, total_size))
                } else {
                    None
                };

                Some((fd, offset, node, shared_region))
            }
            VmBacking::Anonymous { .. } => {
                if effective_shared {
                    // Shared anonymous mappings are not yet represented by a
                    // transferable kernel object.
                    return Err(Errno::EOPNOTSUPP);
                }
                None
            }
        };

        while virt < end {
            let phys =
                if let Some((_, file_offset, _, Some((phys_base, total_size)))) = &file_backing {
                    // 0-copy path
                    let current_offset = *file_offset + (virt - addr as u64);
                    if current_offset >= *total_size as u64 {
                        // Out of bounds for the region
                        return Err(Errno::EINVAL);
                    }
                    phys_base + current_offset
                } else {
                    // Allocation path
                    let allocated_phys = crate::memory::alloc_frame().ok_or(Errno::ENOMEM)?;
                    match req.backing {
                        VmBacking::Anonymous { zeroed } => {
                            if zeroed {
                                let hhdm_virt = allocated_phys + hhdm;
                                unsafe {
                                    core::ptr::write_bytes(hhdm_virt as *mut u8, 0, page_size);
                                }
                            }
                        }
                        VmBacking::File { .. } => {
                            let (file_offset, node) =
                                if let Some((_, file_offset, node, _)) = &file_backing {
                                    (*file_offset, node.clone())
                                } else {
                                    return Err(Errno::EINVAL);
                                };

                            let hhdm_virt = allocated_phys + hhdm;
                            let slice = unsafe {
                                core::slice::from_raw_parts_mut(hhdm_virt as *mut u8, page_size)
                            };

                            let current_offset = file_offset + (virt - addr as u64);
                            let bytes_read = node.read(current_offset, slice)?;

                            if bytes_read < page_size {
                                unsafe {
                                    core::ptr::write_bytes(
                                        (hhdm_virt + bytes_read as u64) as *mut u8,
                                        0,
                                        page_size - bytes_read,
                                    );
                                }
                            }
                        }
                    }
                    allocated_phys
                };

            unsafe {
                crate::memory::map_user_page_with_perms(virt, phys, perms)?;
            }
            virt += page_size as u64;
        }
    }

    let region = VmRegionInfo {
        start: addr,
        end: addr + len,
        prot: req.prot,
        flags: req.flags,
        backing_kind: match req.backing {
            VmBacking::Anonymous { .. } => VmBackingKind::Anonymous,
            VmBacking::File { .. } => VmBackingKind::File,
        },
        _reserved: [0; 7],
    };

    // Only register regions that are actually accessible (not guard-only)
    // Guard regions are virtual address reservations without actual page mappings
    let is_accessible = req.prot.contains(VmProt::READ) || req.prot.contains(VmProt::WRITE);

    if is_accessible {
        // For FIXED mappings, we may be replacing part of an existing region
        // Remove the overlap first to avoid permission conflicts
        if fixed {
            unsafe {
                let _ = crate::sched::remove_user_mappings_current(addr, len);
            }
        }

        unsafe {
            crate::sched::add_user_mapping_current(region)?;
        }
    }

    let resp = VmMapResp { addr, len };
    let resp_slice =
        unsafe { core::slice::from_raw_parts(&resp as *const _ as *const u8, resp_size) };
    unsafe {
        copyout(resp_ptr, resp_slice)?;
    }
    Ok(0)
}

pub fn sys_vm_unmap(req_ptr: usize, resp_ptr: usize) -> SysResult<usize> {
    use crate::syscall::validate::{copyin, copyout, validate_user_range};

    let req_size = core::mem::size_of::<VmUnmapReq>();
    let resp_size = core::mem::size_of::<VmUnmapResp>();
    validate_user_range(req_ptr, req_size, false)?;
    validate_user_range(resp_ptr, resp_size, true)?;

    let mut req: VmUnmapReq = unsafe { core::mem::zeroed() };
    let req_slice =
        unsafe { core::slice::from_raw_parts_mut(&mut req as *mut _ as *mut u8, req_size) };
    unsafe {
        copyin(req_slice, req_ptr)?;
    }

    let page_size = 4096usize;
    let len = align_up(req.len, page_size);
    if len == 0 || req.addr % page_size != 0 {
        return Err(Errno::EINVAL);
    }

    // Update mappings
    let removed_ranges = unsafe { crate::sched::remove_user_mappings_current(req.addr, len)? };

    // Unmap pages
    for (start, end) in removed_ranges {
        let mut virt = start as u64;
        let end_virt = end as u64;
        while virt < end_virt {
            unsafe {
                let _ = crate::memory::unmap_user_page(virt);
            }
            virt += page_size as u64;
        }
    }

    let resp = VmUnmapResp { unmapped_len: len };
    let resp_slice =
        unsafe { core::slice::from_raw_parts(&resp as *const _ as *const u8, resp_size) };
    unsafe {
        copyout(resp_ptr, resp_slice)?;
    }

    Ok(0)
}

pub fn sys_vm_protect(req_ptr: usize) -> SysResult<usize> {
    use crate::syscall::validate::copyin;
    use abi::vm::VmProtectReq;

    let mut req: VmProtectReq = unsafe { core::mem::zeroed() };
    let req_slice = unsafe {
        core::slice::from_raw_parts_mut(
            &mut req as *mut VmProtectReq as *mut u8,
            core::mem::size_of::<VmProtectReq>(),
        )
    };
    unsafe { copyin(req_slice, req_ptr)? };

    // Page-align address and length
    let addr = req.addr as u64 & !0xFFF;
    let len = (req.len + 0xFFF) & !0xFFF;

    crate::sched::hooks::protect_user_range_current(addr, len, req.prot)?;

    Ok(0)
}

pub fn sys_vm_advise(_req_ptr: usize) -> SysResult<usize> {
    Ok(0)
}

pub fn sys_vm_query(req_ptr: usize, resp_ptr: usize) -> SysResult<usize> {
    let _ = (req_ptr, resp_ptr);
    Err(Errno::ENOSYS)
}

pub fn sys_memfd_create(name_ptr: usize, name_len: usize, size: usize) -> SysResult<usize> {
    use crate::syscall::validate::{copyin, validate_user_range};

    // Validate name
    if name_len > 64 {
        return Err(Errno::EINVAL);
    }
    validate_user_range(name_ptr, name_len, false)?;

    let mut name_buf = [0u8; 64];
    unsafe {
        copyin(&mut name_buf[..name_len], name_ptr)?;
    }
    let name = core::str::from_utf8(&name_buf[..name_len]).map_err(|_| Errno::EINVAL)?;

    // Create node
    let node = crate::vfs::memfd::MemFdNode::new(name, size)?;
    let node_arc: alloc::sync::Arc<dyn crate::vfs::VfsNode> = alloc::sync::Arc::new(node);

    // Install in FD table
    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let mut pinfo = pinfo_arc.lock();
    let fd = pinfo.fd_table.open(
        node_arc,
        crate::vfs::OpenFlags::read_write(),
        "memfd".into(),
    )?;
    Ok(fd as usize)
}

pub fn sys_memfd_phys(fd: usize) -> SysResult<usize> {
    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let node = {
        let lock = pinfo_arc.lock();
        let file = lock.fd_table.get(fd as u32)?;
        file.node.clone()
    };

    let (phys, _len) = node.phys_region()?;
    Ok(phys as usize)
}

fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

fn map_perms_from_prot(prot: VmProt) -> crate::MapPerms {
    crate::MapPerms {
        user: prot.contains(VmProt::USER),
        read: prot.contains(VmProt::READ),
        write: prot.contains(VmProt::WRITE),
        exec: prot.contains(VmProt::EXEC),
        kind: crate::MapKind::Normal,
    }
}
