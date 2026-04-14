//! Device capability syscalls

extern crate alloc;

use super::{copyin, copyout};
use crate::syscall::validate::validate_user_range;
use abi::device::{
    DEVICE_IRQ_SUBSCRIBE_DEVICE, DEVICE_IRQ_SUBSCRIBE_VECTOR, DeviceCall, DeviceKind,
    PCI_IRQ_MODE_MSI, PCI_IRQ_MODE_MSIX, PCI_OP_ENABLE_MSI, PciEnableMsiRequest,
    PciEnableMsiResponse,
};
use abi::errors::{Errno, SysResult};

#[repr(C)]
#[derive(Clone, Copy)]
struct RawDeviceCall {
    kind: u32,
    op: u32,
    in_ptr: u64,
    in_len: u32,
    out_ptr: u64,
    out_len: u32,
}

fn decode_device_call(raw: RawDeviceCall) -> SysResult<DeviceCall> {
    let kind = match raw.kind {
        1 => DeviceKind::RtcCmos,
        2 => DeviceKind::Keyboard,
        3 => DeviceKind::Mouse,
        4 => DeviceKind::Framebuffer,
        5 => DeviceKind::Pci,
        6 => DeviceKind::Display,
        7 => DeviceKind::Terminal,
        8 => DeviceKind::Audio,
        _ => return Err(Errno::EINVAL),
    };

    Ok(DeviceCall {
        kind,
        op: raw.op,
        in_ptr: raw.in_ptr,
        in_len: raw.in_len,
        out_ptr: raw.out_ptr,
        out_len: raw.out_len,
    })
}

/// Maximum allowed length for a device sysfs path passed to `SYS_DEVICE_CLAIM`.
const MAX_DEVICE_PATH_LEN: usize = 256;

pub fn sys_device_call(call_ptr: usize) -> SysResult<usize> {
    let size = core::mem::size_of::<RawDeviceCall>();
    validate_user_range(call_ptr, size, true)?;
    let mut raw = RawDeviceCall {
        kind: DeviceKind::Pci as u32,
        op: 0,
        in_ptr: 0,
        in_len: 0,
        out_ptr: 0,
        out_len: 0,
    };
    let slice = unsafe { core::slice::from_raw_parts_mut(&mut raw as *mut _ as *mut u8, size) };
    unsafe {
        copyin(slice, call_ptr)?;
    }
    let call = decode_device_call(raw)?;
    match call.kind {
        DeviceKind::RtcCmos => Err(Errno::NotSupported),
        DeviceKind::Pci => sys_pci_call(&call),
        _ => Err(Errno::NotSupported),
    }
}

pub fn sys_device_claim(path_ptr: usize, path_len: usize) -> SysResult<usize> {
    use crate::device_registry::REGISTRY;

    if path_len == 0 || path_len > MAX_DEVICE_PATH_LEN {
        return Err(Errno::EINVAL);
    }
    validate_user_range(path_ptr, path_len, false)?;
    let mut buf = alloc::vec![0u8; path_len];
    unsafe { copyin(&mut buf, path_ptr)? };
    let full_path = core::str::from_utf8(&buf).map_err(|_| Errno::EINVAL)?;

    // Accept either a full sysfs path (/sys/devices/<slot>) or a bare slot name.
    let slot = full_path
        .trim_start_matches("/sys/devices/")
        .trim_matches('/');

    let task_id = unsafe { crate::sched::current_tid_current() };

    let res = {
        let mut reg = REGISTRY.lock();
        if let Some(device_idx) = reg.find_by_slot(slot) {
            if let Some(claim_handle) = reg.claim(device_idx, task_id) {
                Ok(claim_handle)
            } else {
                Err(Errno::EBUSY)
            }
        } else {
            Err(Errno::ENODEV)
        }
    };

    match res {
        Ok(claim_handle) => {
            crate::kdebug!(
                "DEVICE: task {} claimed device '{}' (handle {})",
                task_id,
                slot,
                claim_handle
            );
            Ok(claim_handle)
        }
        Err(Errno::EBUSY) => {
            crate::kdebug!(
                "DEVICE: claim failed - device '{}' is already claimed",
                slot
            );
            Err(Errno::EBUSY)
        }
        Err(e) => {
            crate::kdebug!("DEVICE: device '{}' not found in registry", slot);
            Err(e)
        }
    }
}

/// Map a device MMIO BAR into the task's address space
pub fn sys_device_map_mmio(claim_handle: usize, bar_index: usize) -> SysResult<usize> {
    use crate::device_registry::REGISTRY;

    if bar_index > 5 {
        return Err(Errno::EINVAL);
    }

    let task_id = unsafe { crate::sched::current_tid_current() };
    let bar_res = {
        let reg = REGISTRY.lock();
        if !reg.verify_claim(claim_handle, task_id) {
            Err(Errno::EPERM)
        } else {
            reg.get_bar_info(claim_handle, bar_index)
                .ok_or(Errno::ENODEV)
        }
    };

    let (phys_addr, size) = match bar_res {
        Ok(info) => info,
        Err(Errno::EPERM) => {
            crate::kdebug!(
                "DEVICE: map_mmio failed - claim {} not owned by task {}",
                claim_handle,
                task_id
            );
            return Err(Errno::EPERM);
        }
        Err(e) => {
            crate::kdebug!(
                "DEVICE: map_mmio failed - BAR{} info not found for claim {}",
                bar_index,
                claim_handle
            );
            return Err(e);
        }
    };

    if phys_addr == 0 || size == 0 {
        crate::kdebug!("DEVICE: map_mmio failed - BAR{} phys/size is 0", bar_index);
        return Err(Errno::ENODEV);
    }

    // Safety check: don't allow mapping more than 1GB in one go to prevent DOS/hangs
    if size > 1024 * 1024 * 1024 {
        crate::kdebug!(
            "DEVICE: map_mmio failed - requested size 0x{:x} exceeds 1GB safety limit",
            size
        );
        return Err(Errno::EINVAL);
    }

    crate::kdebug!(
        "DEVICE: mapping BAR{} (phys=0x{:x}, size=0x{:x}) for task {}",
        bar_index,
        phys_addr,
        size,
        task_id
    );

    let page_count = (size + 4095) / 4096;
    let user_va = crate::memory::alloc_user_va((page_count * 4096) as usize);

    // Map pages WITHOUT holding the registry lock
    for i in 0..page_count {
        let phys = phys_addr as u64 + (i * 4096) as u64;
        let virt = user_va + (i * 4096) as u64;
        unsafe {
            crate::memory::map_user_page_with_perms(
                virt,
                phys,
                crate::MapPerms {
                    user: true,
                    read: true,
                    write: true,
                    exec: false,
                    kind: crate::MapKind::Device,
                },
            )
            .map_err(|_| Errno::ENOMEM)?;
        }
    }

    {
        let mut reg = REGISTRY.lock();
        reg.set_bar_mapping(claim_handle, bar_index, user_va);
    }

    crate::kdebug!(
        "DEVICE: Mapped BAR{} phys=0x{:x} size=0x{:x} -> virt=0x{:x}",
        bar_index,
        phys_addr,
        size,
        user_va
    );

    Ok(user_va as usize)
}

/// Subscribe to device interrupts
///
/// Args (mode=DEVICE_IRQ_SUBSCRIBE_VECTOR):
///   arg0: CPU interrupt vector to subscribe to
///
/// Args (mode=DEVICE_IRQ_SUBSCRIBE_DEVICE):
///   arg0: claim handle
///   arg1: device interrupt index
///
/// Returns: 0 on success
pub fn sys_device_irq_subscribe(arg0: usize, arg1: usize, mode: usize) -> SysResult<usize> {
    match mode as u8 {
        DEVICE_IRQ_SUBSCRIBE_DEVICE => {
            let claim_handle = arg0;
            let irq_index = arg1;
            let task_id = unsafe { crate::sched::current_tid_current() };
            let (irq_mode, vector) = {
                let reg = crate::device_registry::REGISTRY.lock();
                if !reg.verify_claim(claim_handle, task_id) {
                    return Err(Errno::EPERM);
                }
                reg.get_irq_vector(claim_handle, irq_index)
                    .ok_or(Errno::ENODEV)?
            };
            crate::irq::subscribe(vector).map_err(|_| Errno::EBUSY)?;
            crate::kdebug!(
                "DEVICE: task subscribed to device irq {} (mode={:?}, vector=0x{:x})",
                irq_index,
                irq_mode,
                vector
            );
            Ok(0)
        }
        DEVICE_IRQ_SUBSCRIBE_VECTOR | _ => {
            let vector = arg0;
            if vector > 255 {
                return Err(Errno::EINVAL);
            }
            crate::irq::subscribe(vector as u8).map_err(|_| Errno::EBUSY)?;
            crate::kdebug!("DEVICE: task subscribed to vector 0x{:x}", vector);
            Ok(0)
        }
    }
}

/// Wait for a device interrupt
///
/// Args follow sys_device_irq_subscribe
///
/// Returns: number of pending interrupts since last wait
pub fn sys_device_irq_wait(arg0: usize, arg1: usize, mode: usize) -> SysResult<usize> {
    let vector = match mode as u8 {
        DEVICE_IRQ_SUBSCRIBE_DEVICE => {
            let claim_handle = arg0;
            let irq_index = arg1;
            let task_id = unsafe { crate::sched::current_tid_current() };
            let (_mode, vector) = {
                let reg = crate::device_registry::REGISTRY.lock();
                if !reg.verify_claim(claim_handle, task_id) {
                    return Err(Errno::EPERM);
                }
                reg.get_irq_vector(claim_handle, irq_index)
                    .ok_or(Errno::ENODEV)?
            };
            vector
        }
        DEVICE_IRQ_SUBSCRIBE_VECTOR | _ => {
            if arg0 > 255 {
                return Err(Errno::EINVAL);
            }
            arg0 as u8
        }
    };

    let count = crate::irq::wait(vector);
    Ok(count as usize)
}

fn sys_pci_call(call: &DeviceCall) -> SysResult<usize> {
    match call.op {
        PCI_OP_ENABLE_MSI => {
            if call.in_len as usize != core::mem::size_of::<PciEnableMsiRequest>() {
                return Err(Errno::EINVAL);
            }
            if call.out_len as usize != core::mem::size_of::<PciEnableMsiResponse>() {
                return Err(Errno::EINVAL);
            }
            validate_user_range(call.in_ptr as usize, call.in_len as usize, true)?;
            validate_user_range(call.out_ptr as usize, call.out_len as usize, true)?;

            let mut req: PciEnableMsiRequest = unsafe { core::mem::zeroed() };
            let in_slice = unsafe {
                core::slice::from_raw_parts_mut(
                    &mut req as *mut _ as *mut u8,
                    core::mem::size_of::<PciEnableMsiRequest>(),
                )
            };
            unsafe {
                copyin(in_slice, call.in_ptr as usize)?;
            }

            let res = crate::irq::msi::enable_for_claim(
                req.claim_handle as usize,
                req.requested_vectors,
                req.prefer_msix != 0,
            )?;

            let irq_mode = match res.mode {
                crate::device_registry::IrqMode::Msi => PCI_IRQ_MODE_MSI,
                crate::device_registry::IrqMode::Msix => PCI_IRQ_MODE_MSIX,
                crate::device_registry::IrqMode::Legacy => 0,
            };
            let out = PciEnableMsiResponse {
                vector: res.vector,
                irq_mode,
                _reserved: [0; 2],
            };
            let out_slice = unsafe {
                core::slice::from_raw_parts(
                    &out as *const _ as *const u8,
                    core::mem::size_of::<PciEnableMsiResponse>(),
                )
            };
            unsafe {
                copyout(call.out_ptr as usize, out_slice)?;
            }
            Ok(0)
        }
        _ => Err(Errno::NotSupported),
    }
}

/// Allocate DMA-safe memory for a device
pub fn sys_device_alloc_dma(claim_handle: usize, page_count: usize) -> SysResult<usize> {
    use crate::device_registry::REGISTRY;
    use crate::memory::FRAME_ALLOCATOR;

    if page_count == 0 || page_count > 256 {
        return Err(Errno::EINVAL);
    }

    let task_id = unsafe { crate::sched::current_tid_current() };

    {
        let reg = REGISTRY.lock();
        if !reg.verify_claim(claim_handle, task_id) {
            return Err(Errno::EPERM);
        }
    }

    // Allocate contiguous physical frames for DMA
    let phys_base = match crate::memory::alloc_contiguous_frames(page_count) {
        Some(phys) => phys,
        None => {
            crate::kinfo!(
                "DEVICE: DMA alloc failed ({} pages) - no contiguous memory",
                page_count
            );
            return Err(Errno::ENOMEM);
        }
    };

    // Allocate userspace virtual address and map the DMA pages there
    let user_va = crate::memory::alloc_user_va(page_count * 4096);

    for i in 0..page_count {
        let phys = phys_base + (i * 4096) as u64;
        let virt = user_va + (i * 4096) as u64;
        unsafe {
            crate::memory::map_user_page_with_perms(
                virt,
                phys,
                crate::MapPerms {
                    user: true,
                    read: true,
                    write: true,
                    exec: false,
                    kind: crate::MapKind::Device,
                },
            )
            .map_err(|_| Errno::ENOMEM)?;
        }
    }

    // Track the DMA allocation for later phys lookups
    let slot_res = {
        let mut reg = REGISTRY.lock();
        reg.alloc_dma_slot(claim_handle, phys_base, user_va, page_count)
            .ok_or(Errno::ENOMEM)
    };

    if let Err(e) = slot_res {
        crate::kerror!("DEVICE: DMA alloc failed (no slots) for task {}", task_id);
        return Err(e);
    }

    crate::kdebug!(
        "DEVICE: DMA alloc {} pages phys=0x{:x} -> user_va=0x{:x}",
        page_count,
        phys_base,
        user_va
    );

    Ok(user_va as usize)
}

/// Get physical address of a DMA allocation
pub fn sys_device_dma_phys(virt_addr: usize) -> SysResult<usize> {
    let hhdm_offset = crate::boot_info::get().map(|i| i.hhdm_offset).unwrap_or(0);
    if (virt_addr as u64) < hhdm_offset {
        // Try translating referencing current user page table
        if let Some(phys) = crate::memory::translate_user_page(virt_addr as u64) {
            return Ok(phys as usize);
        }

        return Err(Errno::EINVAL);
    }
    let phys = (virt_addr as u64) - hhdm_offset;
    Ok(phys as usize)
}

pub fn sys_device_ioport(port: usize, val: usize, write: bool, width: usize) -> SysResult<usize> {
    if write {
        match width {
            1 => crate::ioport_write_u8(port as u16, val as u8),
            2 => crate::ioport_write_u16(port as u16, val as u16),
            4 => crate::ioport_write_u32(port as u16, val as u32),
            _ => return Err(Errno::EINVAL),
        }
        Ok(0)
    } else {
        let ret = match width {
            1 => crate::ioport_read_u8(port as u16) as usize,
            2 => crate::ioport_read_u16(port as u16) as usize,
            4 => crate::ioport_read_u32(port as u16) as usize,
            _ => return Err(Errno::EINVAL),
        };
        Ok(ret)
    }
}
