#[allow(unused_imports)]
use core::ptr::{read_volatile, write_volatile};

use alloc::string::String;

use abi::errors::Errno;

use crate::device_registry::{IrqMode, MsiCapability, MsixCapability, PciLocation, REGISTRY};
use crate::irq::{alloc_vector, free_vector};
use crate::runtime_base;

pub struct EnableResult {
    pub vector: u8,
    pub mode: IrqMode,
}

pub fn enable_for_claim(
    claim_handle: usize,
    requested_vectors: u16,
    prefer_msix: bool,
) -> Result<EnableResult, Errno> {
    if requested_vectors == 0 || requested_vectors > 1 {
        return Err(Errno::EINVAL);
    }

    let (location, msi_cap, msix_cap, resource_id, bars) = {
        let reg = REGISTRY.lock();
        let (location, msi_cap, msix_cap) = reg.get_pci_info(claim_handle).ok_or(Errno::ENODEV)?;
        let bars = reg.get_bars(claim_handle).ok_or(Errno::ENODEV)?;
        let resource_id = reg
            .get_resource_id_for_claim(claim_handle)
            .ok_or(Errno::ENODEV)?;
        (location, msi_cap, msix_cap, resource_id, bars)
    };

    if resource_id == 0 {
        return Err(Errno::ENODEV);
    }

    let vector = alloc_vector(resource_id, 0).ok_or(Errno::ENOMEM)?;
    let result = if prefer_msix {
        if let Some(msix) = msix_cap {
            program_msix(location, msix, vector, bars)?;
            Ok(EnableResult {
                vector,
                mode: IrqMode::Msix,
            })
        } else if let Some(msi) = msi_cap {
            program_msi(location, msi, vector)?;
            Ok(EnableResult {
                vector,
                mode: IrqMode::Msi,
            })
        } else {
            Err(Errno::ENOSYS)
        }
    } else if let Some(msi) = msi_cap {
        program_msi(location, msi, vector)?;
        Ok(EnableResult {
            vector,
            mode: IrqMode::Msi,
        })
    } else if let Some(msix) = msix_cap {
        program_msix(location, msix, vector, bars)?;
        Ok(EnableResult {
            vector,
            mode: IrqMode::Msix,
        })
    } else {
        Err(Errno::ENOSYS)
    };

    match result {
        Ok(res) => {
            let mut reg = REGISTRY.lock();
            reg.set_irq_mode(claim_handle, res.mode, res.vector);
            Ok(res)
        }
        Err(err) => {
            free_vector(vector);
            Err(err)
        }
    }
}

fn program_msi(location: PciLocation, msi: MsiCapability, vector: u8) -> Result<(), Errno> {
    let (addr, data) = build_msi_message(vector);
    let ctrl = pci_read_config_u16(location, msi.offset + 0x2);
    let mut ctrl_new = ctrl & !0x70;
    ctrl_new |= 0x1;

    unsafe {
        runtime_base()
            .pci_cfg_write32(
                location.bus,
                location.dev,
                location.func,
                msi.offset + 0x4,
                addr,
            )
            .ok();
        if msi.is_64bit {
            runtime_base()
                .pci_cfg_write32(
                    location.bus,
                    location.dev,
                    location.func,
                    msi.offset + 0x8,
                    0,
                )
                .ok();
            pci_write_config_u16(location, msi.offset + 0xC, data as u16);
        } else {
            pci_write_config_u16(location, msi.offset + 0x8, data as u16);
        }
        pci_write_config_u16(location, msi.offset + 0x2, ctrl_new);
    }

    Ok(())
}

fn program_msix(
    location: PciLocation,
    msix: MsixCapability,
    vector: u8,
    bars: ([u64; 6], [u64; 6]),
) -> Result<(), Errno> {
    let (addr, data) = build_msi_message(vector);
    let (bar_addrs, bar_sizes) = bars;
    let bar_index = msix.table_bar as usize;
    if bar_index >= bar_addrs.len() {
        return Err(Errno::EINVAL);
    }
    let bar_base = bar_addrs[bar_index];
    let bar_size = bar_sizes[bar_index];
    if bar_base == 0 || bar_size == 0 {
        return Err(Errno::ENODEV);
    }
    if (msix.table_offset as u64) + 16 > bar_size {
        return Err(Errno::EINVAL);
    }

    let table_phys = bar_base + msix.table_offset as u64;
    let hhdm = crate::boot_info::get().map(|i| i.hhdm_offset).unwrap_or(0);
    let table_virt = table_phys + hhdm;

    unsafe {
        let entry = table_virt as *mut u32;
        write_volatile(entry, addr);
        write_volatile(entry.add(1), 0);
        write_volatile(entry.add(2), data);
        write_volatile(entry.add(3), 0);
    }

    let mut ctrl = pci_read_config_u16(location, msix.offset + 0x2);
    ctrl |= 1 << 15;
    ctrl &= !(1 << 14);
    pci_write_config_u16(location, msix.offset + 0x2, ctrl);

    Ok(())
}

fn build_msi_message(vector: u8) -> (u32, u32) {
    let dest_id = crate::runtime_base().lapic_id().unwrap_or(0);
    let addr = 0xFEE0_0000u32 | (dest_id << 12);
    let data = vector as u32;
    (addr, data)
}

fn pci_read_config_u16(location: PciLocation, offset: u8) -> u16 {
    let aligned = offset & !0x3;
    let shift = (offset & 0x2) * 8;
    let val = runtime_base()
        .pci_cfg_read32(location.bus, location.dev, location.func, aligned)
        .unwrap_or(0);
    ((val >> shift) & 0xFFFF) as u16
}

fn pci_write_config_u16(location: PciLocation, offset: u8, value: u16) {
    let aligned = offset & !0x3;
    let shift = (offset & 0x2) * 8;
    let mut val = runtime_base()
        .pci_cfg_read32(location.bus, location.dev, location.func, aligned)
        .unwrap_or(0);
    val &= !(0xFFFF << shift);
    val |= (value as u32) << shift;
    runtime_base()
        .pci_cfg_write32(location.bus, location.dev, location.func, aligned, val)
        .ok();
}

fn update_device_irq(_resource_id: u64, _mode: IrqMode, _vector: u8) {
    // Device IRQ state is tracked via the registry, not the graph.
}
