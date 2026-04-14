//! Device Registry for capability-based device claiming
//!
//! This module tracks claimable devices and their allowed I/O port ranges
//! and MMIO BARs. When a task claims a device, it receives a handle that
//! authorizes resource access within the device's declared ranges.

use spin::Mutex;

/// Maximum number of devices in the registry
const MAX_DEVICES: usize = 16;

/// Maximum number of claimed devices across all tasks
const MAX_CLAIMS: usize = 32;

/// Maximum BARs per device
const MAX_BARS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrqMode {
    Legacy = 0,
    Msi = 1,
    Msix = 2,
}

#[derive(Clone, Copy, Default)]
pub struct PciLocation {
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
}

#[derive(Clone, Copy, Default)]
pub struct MsiCapability {
    pub offset: u8,
    pub is_64bit: bool,
    pub has_mask: bool,
}

#[derive(Clone, Copy, Default)]
pub struct MsixCapability {
    pub offset: u8,
    pub table_bar: u8,
    pub table_offset: u32,
}

/// A device entry in the registry
#[derive(Clone, Copy)]
pub struct DeviceEntry {
    pub kind: &'static str,
    pub ioport_ranges: &'static [(u16, u16)], // (start, end) inclusive
    pub resource_id: u64,                     // Kernel resource ID for this device
    pub mmio_bars: [u64; MAX_BARS],           // BAR physical addresses
    pub mmio_sizes: [u64; MAX_BARS],          // BAR sizes
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u8,
    pub subclass: u8,
    pub prog_if: u8,
    pub pci_location: Option<PciLocation>,
    pub msi_cap: Option<MsiCapability>,
    pub msix_cap: Option<MsixCapability>,
    pub irq_mode: IrqMode,
    pub irq_vector: u8,
}

impl DeviceEntry {
    pub const fn new_legacy(
        kind: &'static str,
        ioport_ranges: &'static [(u16, u16)],
        resource_id: u64,
    ) -> Self {
        Self {
            kind,
            ioport_ranges,
            resource_id,
            mmio_bars: [0; MAX_BARS],
            mmio_sizes: [0; MAX_BARS],
            vendor_id: 0,
            device_id: 0,
            class_code: 0,
            subclass: 0,
            prog_if: 0,
            pci_location: None,
            msi_cap: None,
            msix_cap: None,
            irq_mode: IrqMode::Legacy,
            irq_vector: 0,
        }
    }

    pub const fn new_mmio(
        kind: &'static str,
        resource_id: u64,
        bars: [u64; MAX_BARS],
        sizes: [u64; MAX_BARS],
    ) -> Self {
        Self {
            kind,
            ioport_ranges: &[],
            resource_id,
            mmio_bars: bars,
            mmio_sizes: sizes,
            vendor_id: 0,
            device_id: 0,
            class_code: 0,
            subclass: 0,
            prog_if: 0,
            pci_location: None,
            msi_cap: None,
            msix_cap: None,
            irq_mode: IrqMode::Legacy,
            irq_vector: 0,
        }
    }
}

/// DMA buffer allocation for a claim
#[derive(Clone, Copy, Default)]
pub struct DmaBuffer {
    pub phys_addr: u64,
    pub virt_addr: u64,
    pub page_count: usize,
    pub valid: bool,
}

/// A claimed device
#[derive(Clone, Copy, Default)]
pub struct ClaimedDevice {
    pub device_index: usize,
    pub task_id: u64,
    pub valid: bool,
    pub mapped_bar_virt: [u64; MAX_BARS], // Virtual addresses of mapped BARs
    pub dma_buffers: [DmaBuffer; 4],      // Up to 4 DMA buffers per claim
}

/// Global device registry
pub static REGISTRY: Mutex<DeviceRegistry> = Mutex::new(DeviceRegistry::new());

pub struct DeviceRegistry {
    devices: [Option<DeviceEntry>; MAX_DEVICES],
    device_count: usize,
    claims: [ClaimedDevice; MAX_CLAIMS],
}

impl DeviceRegistry {
    pub const fn new() -> Self {
        Self {
            devices: [None; MAX_DEVICES],
            device_count: 0,
            claims: [ClaimedDevice {
                device_index: 0,
                task_id: 0,
                valid: false,
                mapped_bar_virt: [0; MAX_BARS],
                dma_buffers: [DmaBuffer {
                    phys_addr: 0,
                    virt_addr: 0,
                    page_count: 0,
                    valid: false,
                }; 4],
            }; MAX_CLAIMS],
        }
    }

    /// Register a device in the registry. Returns device index.
    pub fn register(&mut self, entry: DeviceEntry) -> Option<usize> {
        if self.device_count >= MAX_DEVICES {
            return None;
        }
        let idx = self.device_count;
        self.devices[idx] = Some(entry);
        self.device_count += 1;
        Some(idx)
    }

    pub fn set_pci_info(
        &mut self,
        device_index: usize,
        location: PciLocation,
        msi_cap: Option<MsiCapability>,
        msix_cap: Option<MsixCapability>,
    ) -> bool {
        if device_index >= self.device_count {
            return false;
        }
        if let Some(device) = self.devices[device_index].as_mut() {
            device.pci_location = Some(location);
            device.msi_cap = msi_cap;
            device.msix_cap = msix_cap;
            return true;
        }
        false
    }

    pub fn set_pci_identity(
        &mut self,
        device_index: usize,
        vendor_id: u16,
        device_id: u16,
        class_code: u8,
        subclass: u8,
        prog_if: u8,
    ) -> bool {
        if device_index >= self.device_count {
            return false;
        }
        if let Some(device) = self.devices[device_index].as_mut() {
            device.vendor_id = vendor_id;
            device.device_id = device_id;
            device.class_code = class_code;
            device.subclass = subclass;
            device.prog_if = prog_if;
            return true;
        }
        false
    }

    pub fn len(&self) -> usize {
        self.device_count
    }

    pub fn entry_copy(&self, index: usize) -> Option<DeviceEntry> {
        if index < self.device_count {
            self.devices[index]
        } else {
            None
        }
    }

    /// Get device by index
    pub fn get(&self, index: usize) -> Option<&DeviceEntry> {
        if index < self.device_count {
            self.devices[index].as_ref()
        } else {
            None
        }
    }

    /// Find device by PCI slot name (e.g. `"pci-0000:00:1f.2"`).
    ///
    /// The slot name is derived from the device's [`PciLocation`] in the same
    /// way that sysfs does, so callers can use a sysfs path as the primary key.
    pub fn find_by_slot(&self, slot: &str) -> Option<usize> {
        for i in 0..self.device_count {
            if let Some(entry) = &self.devices[i] {
                if let Some(loc) = entry.pci_location {
                    let name = alloc::format!(
                        "pci-0000:{:02x}:{:02x}.{}",
                        loc.bus, loc.dev, loc.func
                    );
                    if name == slot {
                        return Some(i);
                    }
                }
            }
        }
        None
    }

    /// Find device by resource ID.
    pub fn find_by_resource_id(&self, resource_id: u64) -> Option<usize> {
        for i in 0..self.device_count {
            if let Some(entry) = &self.devices[i] {
                if entry.resource_id == resource_id {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Claim a device for a task. Returns claim handle (index) or None if already claimed.
    pub fn claim(&mut self, device_index: usize, task_id: u64) -> Option<usize> {
        // Check device exists
        if device_index >= self.device_count || self.devices[device_index].is_none() {
            return None;
        }

        // Check not already claimed
        for claim in &self.claims {
            if claim.valid && claim.device_index == device_index {
                // Already claimed
                return None;
            }
        }

        // Find free claim slot
        for (i, claim) in self.claims.iter_mut().enumerate() {
            if !claim.valid {
                claim.device_index = device_index;
                claim.task_id = task_id;
                claim.valid = true;
                claim.mapped_bar_virt = [0; MAX_BARS];
                claim.dma_buffers = [DmaBuffer::default(); 4];
                return Some(i);
            }
        }

        None // No free slots
    }

    /// Get BAR info for a claimed device
    pub fn get_bar_info(&self, claim_handle: usize, bar_index: usize) -> Option<(u64, u64)> {
        if claim_handle >= MAX_CLAIMS || bar_index >= MAX_BARS {
            return None;
        }
        let claim = &self.claims[claim_handle];
        if !claim.valid {
            return None;
        }

        if let Some(device) = self.get(claim.device_index) {
            let addr = device.mmio_bars[bar_index];
            let size = device.mmio_sizes[bar_index];
            if addr != 0 && size != 0 {
                return Some((addr, size));
            }
        }
        None
    }

    /// Record a BAR mapping for a claim
    pub fn set_bar_mapping(&mut self, claim_handle: usize, bar_index: usize, virt_addr: u64) {
        if claim_handle < MAX_CLAIMS && bar_index < MAX_BARS {
            self.claims[claim_handle].mapped_bar_virt[bar_index] = virt_addr;
        }
    }

    pub fn get_pci_info(
        &self,
        claim_handle: usize,
    ) -> Option<(PciLocation, Option<MsiCapability>, Option<MsixCapability>)> {
        if claim_handle >= MAX_CLAIMS {
            return None;
        }
        let claim = &self.claims[claim_handle];
        if !claim.valid {
            return None;
        }
        let device = self.get(claim.device_index)?;
        let location = device.pci_location?;
        Some((location, device.msi_cap, device.msix_cap))
    }

    pub fn get_bars(&self, claim_handle: usize) -> Option<([u64; MAX_BARS], [u64; MAX_BARS])> {
        if claim_handle >= MAX_CLAIMS {
            return None;
        }
        let claim = &self.claims[claim_handle];
        if !claim.valid {
            return None;
        }
        let device = self.get(claim.device_index)?;
        Some((device.mmio_bars, device.mmio_sizes))
    }

    pub fn set_irq_mode(&mut self, claim_handle: usize, mode: IrqMode, vector: u8) -> bool {
        if claim_handle >= MAX_CLAIMS {
            return false;
        }
        let claim = &self.claims[claim_handle];
        if !claim.valid {
            return false;
        }
        if let Some(device) = self.devices[claim.device_index].as_mut() {
            device.irq_mode = mode;
            device.irq_vector = vector;
            return true;
        }
        false
    }

    pub fn get_irq_vector(&self, claim_handle: usize, irq_index: usize) -> Option<(IrqMode, u8)> {
        if claim_handle >= MAX_CLAIMS || irq_index != 0 {
            return None;
        }
        let claim = &self.claims[claim_handle];
        if !claim.valid {
            return None;
        }
        let device = self.get(claim.device_index)?;
        if device.irq_vector == 0 {
            return None;
        }
        Some((device.irq_mode, device.irq_vector))
    }

    pub fn get_resource_id_for_claim(&self, claim_handle: usize) -> Option<u64> {
        if claim_handle >= MAX_CLAIMS {
            return None;
        }
        let claim = &self.claims[claim_handle];
        if !claim.valid {
            return None;
        }
        let device = self.get(claim.device_index)?;
        Some(device.resource_id)
    }

    /// Allocate DMA buffer tracking slot
    pub fn alloc_dma_slot(
        &mut self,
        claim_handle: usize,
        phys: u64,
        virt: u64,
        pages: usize,
    ) -> Option<usize> {
        if claim_handle >= MAX_CLAIMS {
            return None;
        }
        let claim = &mut self.claims[claim_handle];
        if !claim.valid {
            return None;
        }

        for (i, buf) in claim.dma_buffers.iter_mut().enumerate() {
            if !buf.valid {
                buf.phys_addr = phys;
                buf.virt_addr = virt;
                buf.page_count = pages;
                buf.valid = true;
                return Some(i);
            }
        }
        None
    }

    /// Check if a port access is authorized for a given claim handle
    pub fn check_port_access(&self, claim_handle: usize, port: u16) -> bool {
        if claim_handle >= MAX_CLAIMS {
            return false;
        }
        let claim = &self.claims[claim_handle];
        if !claim.valid {
            return false;
        }

        if let Some(device) = self.get(claim.device_index) {
            for &(start, end) in device.ioport_ranges {
                if port >= start && port <= end {
                    return true;
                }
            }
        }
        false
    }

    /// Get claim by handle and verify task ownership
    pub fn verify_claim(&self, claim_handle: usize, task_id: u64) -> bool {
        if claim_handle >= MAX_CLAIMS {
            return false;
        }
        let claim = &self.claims[claim_handle];
        let ok = claim.valid && claim.task_id == task_id;
        ok
    }

    /// Release a claim
    pub fn release(&mut self, claim_handle: usize, task_id: u64) -> bool {
        if claim_handle >= MAX_CLAIMS {
            return false;
        }
        let claim = &mut self.claims[claim_handle];
        if claim.valid && claim.task_id == task_id {
            claim.valid = false;
            true
        } else {
            false
        }
    }

    /// Release all claims owned by a task. Returns the number of claims released.
    pub fn release_all_for_task(&mut self, task_id: u64) -> usize {
        let mut count = 0;
        for claim in self.claims.iter_mut() {
            if claim.valid && claim.task_id == task_id {
                claim.valid = false;
                count += 1;
            }
        }
        count
    }
}

// Static device definitions for legacy devices
pub static CMOS_IOPORT_RANGES: &[(u16, u16)] = &[(0x70, 0x71)];
pub static PS2_IOPORT_RANGES: &[(u16, u16)] = &[(0x60, 0x64)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_claiming_ownership() {
        let mut reg = DeviceRegistry::new();
        let dev_idx = reg
            .register(DeviceEntry::new_legacy("test_dev", &[], 123))
            .unwrap();

        // Task A claims device
        let claim_a = reg
            .claim(dev_idx, 10)
            .expect("Task A should be able to claim");
        assert_eq!(reg.claims[claim_a].task_id, 10);

        // Task B tries to claim same device -> should fail
        let claim_b = reg.claim(dev_idx, 20);
        assert!(
            claim_b.is_none(),
            "Task B should NOT be able to claim already claimed device"
        );

        // Task A exits -> release all
        reg.release_all_for_task(10);
        assert!(
            !reg.claims[claim_a].valid,
            "Claim should be invalid after release"
        );

        // Task C claims same device -> should succeed
        let _claim_c = reg
            .claim(dev_idx, 30)
            .expect("Task C should be able to claim after Task A release");
    }

    #[test]
    fn test_multiple_devices_per_task() {
        let mut reg = DeviceRegistry::new();
        let dev1 = reg
            .register(DeviceEntry::new_legacy("dev1", &[], 1))
            .unwrap();
        let dev2 = reg
            .register(DeviceEntry::new_legacy("dev2", &[], 2))
            .unwrap();

        reg.claim(dev1, 100).unwrap();
        reg.claim(dev2, 100).unwrap();

        reg.release_all_for_task(100);

        for claim in &reg.claims {
            if claim.valid {
                assert_ne!(claim.task_id, 100, "No claims for task 100 should be valid");
            }
        }
    }
}
