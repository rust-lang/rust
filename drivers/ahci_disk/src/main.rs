//! AHCI (SATA) Disk Driver (v2)
//!
//! Userspace driver for AHCI SATA controllers. Detects SATA devices via
//! PCI enumeration and MMIO access.
//!
//! Features:
//! - AHCI Controller Initialization and Port Probing
//! - SATA Disk Registration as Block Devices
//! - SATAPI (CD-ROM) support via PACKET commands
//! - Block Device RPC Service (port-based, headless operation)
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use alloc::vec::Vec;
use core::time::Duration;
use stem::abi::block_device_protocol::*;
use stem::abi::module_manifest::{ManifestHeader, ModuleKind, MANIFEST_MAGIC};
use stem::block::{BlockDevice, BlockError};
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read, vfs_readdir};
use stem::syscall::{channel_create, channel_recv, channel_send, channel_wait, ChannelHandle};
use stem::{debug, error, info};

#[unsafe(link_section = ".thing_manifest")]
#[unsafe(no_mangle)]
#[used]
pub static MANIFEST: ManifestHeader = ManifestHeader {
    magic: MANIFEST_MAGIC,
    kind: ModuleKind::Driver,
    device_kind: *b"dev.storage.Ahci\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
    version: 1,
    _reserved: 0,
};

const PCI_CLASS_STORAGE: u64 = 0x01;
const PCI_SUBCLASS_SATA: u64 = 0x06;
const PCI_PROGIF_AHCI: u64 = 0x01;

const HBA_CAP: usize = 0x00;
const HBA_GHC: usize = 0x04;
const HBA_PI: usize = 0x0C;
const HBA_VS: usize = 0x10;
const HBA_PORT_BASE: usize = 0x100;

const PORT_CLB: usize = 0x00;
const PORT_CLBU: usize = 0x04;
const PORT_FB: usize = 0x08;
const PORT_FBU: usize = 0x0C;
const PORT_IS: usize = 0x10;
const PORT_IE: usize = 0x14;
const PORT_CMD: usize = 0x18;
const PORT_TFD: usize = 0x20;
const PORT_SIG: usize = 0x24;
const PORT_SSTS: usize = 0x28;
const PORT_SCTL: usize = 0x2C;
const PORT_SERR: usize = 0x30;
const PORT_CI: usize = 0x38;

const SATA_SIG_ATA: u32 = 0x00000101;
const SATA_SIG_ATAPI: u32 = 0xEB140101;
const SATA_SIG_SEMB: u32 = 0xC33C0101;
const SATA_SIG_PM: u32 = 0x96690101;

const PORT_CMD_ST: u32 = 1 << 0;
const PORT_CMD_FRE: u32 = 1 << 4;
const PORT_CMD_FR: u32 = 1 << 14;
const PORT_CMD_CR: u32 = 1 << 15;
const PORT_CMD_ATAPI: u32 = 1 << 24;

// Port buffer size for RPC communication
const PORT_BUFFER_SIZE: usize = 4096;
// Maximum sectors per read to fit in port buffer
// Response format: 1 byte (response type) + 4 bytes (ReadResponse) + sector data
// For 512-byte sectors: (4096 - 5) / 512 = 7 sectors max
// For 2048-byte sectors: (4096 - 5) / 2048 = 1 sector max
const MAX_SECTORS_PER_READ_512: u32 = 7;
const MAX_SECTORS_PER_READ_2048: u32 = 1;

const SSTS_DET_MASK: u32 = 0x0F;
const SSTS_DET_PRESENT: u32 = 0x03;
const SSTS_IPM_MASK: u32 = 0x0F00;
const SSTS_IPM_ACTIVE: u32 = 0x0100;

const FIS_TYPE_REG_H2D: u8 = 0x27;
const ATA_CMD_PACKET: u8 = 0xA0;

#[repr(C, align(4096))]
struct DmaBuffer {
    data: [u8; 4096],
}
static mut DMA_BUFFER: DmaBuffer = DmaBuffer { data: [0; 4096] };

struct AhciPort {
    port_num: u32,
    sector_count: u64,
    sector_size: u32,
    supports_lba48: bool,
    model: [u8; 40],
    serial: [u8; 20],
    read_port_handle: Option<ChannelHandle>,
    mmio_base: u64,
    dma_virt: u64,
    dma_phys: u64,
}

// Memory Layout in DMA Buffer (1 page = 4096 bytes)
// 0x000 - 0x400: Command List (32 * 32 = 1024 bytes)
// 0x400 - 0x500: Received FIS (256 bytes)
// 0x500 - 0x580: Command Table (128 bytes)
// 0x600 - 0xE00: Data Buffer (2048 bytes = 1 CDROM sector)
const OFFSET_CMD_LIST: usize = 0x000;
const OFFSET_FIS: usize = 0x400;
const OFFSET_CMD_TABLE: usize = 0x500;
const OFFSET_DATA: usize = 0x600;
const DATA_SIZE: usize = 2048;

#[repr(C, packed)]
struct CommandHeader {
    // DW0
    cfl: u8,    // Command FIS length in DWORDS (5 for H2D)
    pm: u8,     // Port Multiplier
    prdtl: u16, // PRDT Length
    // DW1
    prdbc: u32, // PRD Byte Count
    // DW2, 3
    ctba: u32,  // Command Table Base Address
    ctbau: u32, // Upper 32-bits
    // DW4-7
    reserved: [u32; 4],
}

#[repr(C, packed)]
struct CommandTable {
    // 0x00
    cfis: [u8; 64], // Command FIS
    // 0x40
    acmd: [u8; 16], // ATAPI Command (SCSI CDB)
    // 0x50
    reserved: [u8; 48],
    // 0x80
    prdt: [PrdtEntry; 1], // 1 entry for our simple needs
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct PrdtEntry {
    dba: u32,  // Data Base Address
    dbau: u32, // Upper
    reserved: u32,
    dbc: u32, // Data Byte Count (bit 31 = Interrupt on Completion)
}

fn mmio_read32(base: u64, offset: usize) -> u32 {
    unsafe { core::ptr::read_volatile((base as usize + offset) as *const u32) }
}

fn mmio_write32(base: u64, offset: usize, val: u32) {
    unsafe { core::ptr::write_volatile((base as usize + offset) as *mut u32, val) }
}

fn port_base(hba_base: u64, port: u32) -> u64 {
    hba_base + HBA_PORT_BASE as u64 + (port as u64 * 0x80)
}

fn check_port_type(hba_base: u64, port: u32) -> Option<u32> {
    let pb = port_base(hba_base, port);
    let ssts = mmio_read32(pb, PORT_SSTS);
    let det = ssts & SSTS_DET_MASK;
    let ipm = ssts & SSTS_IPM_MASK;
    if det != SSTS_DET_PRESENT || ipm != SSTS_IPM_ACTIVE {
        return None;
    }
    Some(mmio_read32(pb, PORT_SIG))
}

fn stop_port(hba_base: u64, port: u32) {
    let pb = port_base(hba_base, port);
    let mut cmd = mmio_read32(pb, PORT_CMD);
    cmd &= !PORT_CMD_ST;
    cmd &= !PORT_CMD_FRE;
    mmio_write32(pb, PORT_CMD, cmd);

    // Wait for bits to clear
    for _ in 0..1000 {
        let cmd = mmio_read32(pb, PORT_CMD);
        if cmd & PORT_CMD_CR == 0 && cmd & PORT_CMD_FR == 0 {
            break;
        }
    }
}

fn start_port(hba_base: u64, port: u32) {
    let pb = port_base(hba_base, port);
    // Loop wait logic omitted for brevity, assuming stopped
    let mut cmd = mmio_read32(pb, PORT_CMD);
    cmd |= PORT_CMD_FRE;
    mmio_write32(pb, PORT_CMD, cmd);
    cmd |= PORT_CMD_ST;
    mmio_write32(pb, PORT_CMD, cmd);
}

// -----------------------------------------------------------------------------
// ATAPI Block Device Implementation
// -----------------------------------------------------------------------------

struct AhciAtapiDevice {
    mmio_base: u64,
    port: u32,
    dma_virt: u64,
    dma_phys: u64,
}

impl AhciAtapiDevice {
    fn read_packet(&self, lba: u64, count: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let pb = port_base(self.mmio_base, self.port);

        // Clear interrupt status
        mmio_write32(pb, PORT_IS, 0xFFFFFFFF);

        // Slot 0
        let slot = 0;

        // Prepare Command Table
        let cmd_tbl_addr = (self.dma_virt as usize + OFFSET_CMD_TABLE) as *mut CommandTable;
        let ds = 2048; // Reading 1 sector
        let prdt_entry = PrdtEntry {
            dba: (self.dma_phys as usize + OFFSET_DATA) as u32,
            dbau: ((self.dma_phys as usize + OFFSET_DATA) >> 32) as u32,
            reserved: 0,
            dbc: (ds - 1) as u32, // Byte count - 1
        };

        unsafe {
            let tbl = &mut *cmd_tbl_addr;
            // Setup FIS (RegH2D)
            tbl.cfis[0] = FIS_TYPE_REG_H2D;
            tbl.cfis[1] = 0x80; // Command (Bit 7)
            tbl.cfis[2] = ATA_CMD_PACKET;
            tbl.cfis[3] = 1; // Feature (Bit 0 = DMA)
            tbl.cfis[4] = 0; // LBA Low
            tbl.cfis[5] = (ds as u32 & 0xFF) as u8; // LBA Mid (Byte Count Low)
            tbl.cfis[6] = ((ds as u32 >> 8) & 0xFF) as u8; // LBA High (Byte Count High)
            tbl.cfis[7] = 0; // Device
                             // Rest 0

            // Setup ATAPI CDB (SCSI READ10)
            // Clear ACDB first to avoid garbage
            tbl.acmd = [0; 16];

            let lba32 = lba as u32;
            let count32 = 1u32; // Always read 1 sector here
            tbl.acmd[0] = 0x28; // READ(10)
            tbl.acmd[1] = 0;
            tbl.acmd[2] = (lba32 >> 24) as u8;
            tbl.acmd[3] = (lba32 >> 16) as u8;
            tbl.acmd[4] = (lba32 >> 8) as u8;
            tbl.acmd[5] = lba32 as u8;
            tbl.acmd[6] = 0; // Reserved
            tbl.acmd[7] = (count32 >> 8) as u8; // Length MSB
            tbl.acmd[8] = count32 as u8; // Length LSB
            tbl.acmd[9] = 0; // Control

            tbl.prdt[0] = prdt_entry;
        }

        // Prepare Command Header
        let cmd_hdr_addr = (self.dma_virt as usize + OFFSET_CMD_LIST) as *mut CommandHeader;
        unsafe {
            let hdr = &mut *cmd_hdr_addr.add(slot);
            hdr.cfl = 5; // 5 Dwords for FIS
            hdr.pm = 0;
            hdr.prdtl = 1; // 1 PRDT entry

            // Set Atapi bit (Bit 5 of Byte 0)
            let mut opts = 5u8;
            opts |= 0x20; // ATAPI
            hdr.cfl = opts;

            hdr.prdbc = 0;
            hdr.ctba = (self.dma_phys as usize + OFFSET_CMD_TABLE) as u32;
            hdr.ctbau = ((self.dma_phys as usize + OFFSET_CMD_TABLE) >> 32) as u32;
        }

        // Issue Command via CI (Command Issue)
        // Wait for port to be idle?
        let mut timeout = 100000;
        while (mmio_read32(pb, PORT_TFD) & 0x88) != 0 && timeout > 0 {
            timeout -= 1;
        }
        if timeout == 0 {
            return Err(BlockError::NotReady);
        }

        mmio_write32(pb, PORT_CI, 1 << slot);

        // Wait for completion
        loop {
            let ci = mmio_read32(pb, PORT_CI);
            if (ci & (1 << slot)) == 0 {
                break; // Done
            }
            if (mmio_read32(pb, PORT_IS) & (1 << 30)) != 0 {
                let tfd = mmio_read32(pb, PORT_TFD);
                let serr = mmio_read32(pb, PORT_SERR);
                info!(
                    "AHCI: IoError on port {}. TFD=0x{:x} (STS=0x{:x} ERR=0x{:x}) SERR=0x{:x}",
                    self.port,
                    tfd,
                    tfd & 0xFF,
                    (tfd >> 8) & 0xFF,
                    serr
                );

                // Clear error
                mmio_write32(pb, PORT_SERR, serr);

                return Err(BlockError::IoError);
            }
        }

        // Copy data to user buffer
        let src = unsafe {
            core::slice::from_raw_parts((self.dma_virt as usize + OFFSET_DATA) as *const u8, 2048)
        };
        buf[0..2048].copy_from_slice(src);

        Ok(())
    }
}

impl BlockDevice for AhciAtapiDevice {
    fn read_sectors(&self, lba: u64, count: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if count == 0 {
            return Ok(());
        }
        let mut current_lba = lba;
        let mut buf_offset = 0;

        for _ in 0..count {
            if buf_offset + 2048 > buf.len() {
                return Err(BlockError::InvalidParam);
            }
            // Read 1 sector at a time
            self.read_packet(current_lba, 1, &mut buf[buf_offset..buf_offset + 2048])?;
            current_lba += 1;
            buf_offset += 2048;
        }
        Ok(())
    }

    fn sector_size(&self) -> u64 {
        2048
    }
}

// -----------------------------------------------------------------------------
// ISO Logic (Ported from iso_reader)
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Block Device Registration
// -----------------------------------------------------------------------------

fn register_disk(port: &mut AhciPort) {
    // Create RPC port for block device service (4KB buffer)
    let (write_handle, read_handle) = match channel_create(PORT_BUFFER_SIZE) {
        Ok(handles) => handles,
        Err(e) => {
            error!("AHCI: Failed to create port: {:?}", e);
            return;
        }
    };
    port.read_port_handle = Some(read_handle);

    // Publish to VFS
    use stem::syscall::vfs::{vfs_close, vfs_mkdir, vfs_open, vfs_write};
    let _ = vfs_mkdir("/services/storage");
    let name = alloc::format!("/services/storage/ahci{}", port.port_num);
    if let Ok(fd) = vfs_open(
        &name,
        abi::syscall::vfs_flags::O_CREAT | abi::syscall::vfs_flags::O_RDWR,
    ) {
        let _ = vfs_write(fd, alloc::format!("{}", write_handle).as_bytes());
        let _ = vfs_close(fd);
    }

    let model_str = core::str::from_utf8(&port.model)
        .unwrap_or("Unknown")
        .trim();

    info!(
        "AHCI: Registered block device port={} sectors={} lba48={} model='{}' rpc_port={}",
        port.port_num, port.sector_count, port.supports_lba48, model_str, write_handle
    );
}

fn register_atapi_disk(port: &mut AhciPort) {
    // Create RPC port for block device service (4KB buffer)
    let (write_handle, read_handle) = match channel_create(PORT_BUFFER_SIZE) {
        Ok(handles) => handles,
        Err(e) => {
            error!("AHCI: Failed to create port: {:?}", e);
            return;
        }
    };
    port.read_port_handle = Some(read_handle);

    // Publish to VFS
    use stem::syscall::vfs::{vfs_close, vfs_mkdir, vfs_open, vfs_write};
    let _ = vfs_mkdir("/services/storage");
    let name = alloc::format!("/services/storage/atapi{}", port.port_num);
    if let Ok(fd) = vfs_open(
        &name,
        abi::syscall::vfs_flags::O_CREAT | abi::syscall::vfs_flags::O_RDWR,
    ) {
        let _ = vfs_write(fd, alloc::format!("{}", write_handle).as_bytes());
        let _ = vfs_close(fd);
    }

    let model_str = core::str::from_utf8(&port.model)
        .unwrap_or("ATAPI Device")
        .trim();

    debug!(
        "AHCI: Registered ATAPI block device port={} model='{}' rpc_port={}",
        port.port_num, model_str, write_handle
    );
}

#[stem::main]
fn main(boot_fd: usize) -> ! {
    debug!("AHCI: Starting AHCI/SATA disk driver (boot_fd={})", boot_fd);

    // 1. Get device path from argv or primary arg
    let mut boot_fd = boot_fd;
    if boot_fd == 0 {
        let mut buf = [0u8; 1024];
        if let Ok(needed) = stem::syscall::argv_get(&mut buf) {
            if needed >= 4 {
                let count = u32::from_le_bytes(buf[0..4].try_into().unwrap());
                if count >= 2 {
                    let mut offset = 4;
                    let arg0_len =
                        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4 + arg0_len;
                    if offset + 4 <= buf.len() {
                        let arg1_len =
                            u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
                                as usize;
                        offset += 4;
                        if offset + arg1_len <= buf.len() {
                            if let Ok(s) = core::str::from_utf8(&buf[offset..offset + arg1_len]) {
                                if let Ok(val) = s.parse::<usize>() {
                                    boot_fd = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut path_buf = [0u8; 128];
    let path = if boot_fd != 0 {
        use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
        let req = VmMapReq {
            addr_hint: 0,
            len: 4096,
            prot: VmProt::READ | VmProt::USER,
            flags: VmMapFlags::empty(),
            backing: VmBacking::File {
                fd: boot_fd as u32,
                offset: 0,
            },
        };
        if let Ok(resp) = stem::syscall::vm_map(&req) {
            let ptr = resp.addr as *const u8;
            let len = (0..128)
                .find(|&i| unsafe { *ptr.add(i) == 0 })
                .unwrap_or(128);
            unsafe { core::slice::from_raw_parts(ptr, len) }
        } else {
            b"/sys/devices/pci-0000:00:1f.2" // Default to QEMU AHCI (q35)
        }
    } else {
        b"/sys/devices/pci-0000:00:1f.2"
    };
    let path_str = core::str::from_utf8(path).unwrap_or("");

    // Resolve the sysfs path for this device.
    let dev_path = if !path_str.is_empty() {
        alloc::string::String::from(path_str)
    } else {
        match find_ahci_device() {
            Some(p) => p,
            None => {
                error!("AHCI: Failed to find controller info at '{}'", path_str);
                loop {
                    stem::sleep(Duration::from_secs(60));
                }
            }
        }
    };

    let claim_handle = match stem::syscall::device_claim(&dev_path) {
        Ok(h) => {
            debug!("AHCI: Claimed PCI device '{}' handle={}", dev_path, h);
            h
        }
        Err(e) => {
            error!("AHCI: Failed to claim '{}': {:?}", dev_path, e);
            loop {
                stem::sleep(Duration::from_secs(60));
            }
        }
    };

    let mapped_base = match stem::syscall::device_map_mmio(claim_handle, 5) {
        Ok(addr) => {
            debug!("AHCI: Mapped ABAR at 0x{:x}", addr);
            addr
        }
        Err(e) => {
            error!("AHCI: Failed to map MMIO: {:?}", e);
            loop {
                stem::sleep(Duration::from_secs(60));
            }
        }
    };

    let cap = mmio_read32(mapped_base, HBA_CAP);
    let version = mmio_read32(mapped_base, HBA_VS);
    let pi = mmio_read32(mapped_base, HBA_PI);
    debug!(
        "AHCI: Version {}.{}, {} ports, {} slots, 64-bit: {}",
        (version >> 16) & 0xFFFF,
        version & 0xFFFF,
        ((cap >> 0) & 0x1F) + 1,
        ((cap >> 8) & 0x1F) + 1,
        (cap & (1 << 31)) != 0
    );
    debug!("AHCI: Ports implemented: 0x{:x}", pi);

    // Enable AHCI mode
    let mut ghc = mmio_read32(mapped_base, HBA_GHC);
    ghc |= 1 << 31;
    mmio_write32(mapped_base, HBA_GHC, ghc);

    // Use static buffer for DMA to avoid kernel address issues
    let dma_virt = unsafe { DMA_BUFFER.data.as_mut_ptr() as u64 };
    debug!("AHCI: DMA virt=0x{:x}", dma_virt);

    let dma_phys = match stem::syscall::device_dma_phys(dma_virt) {
        Ok(addr) => {
            debug!("AHCI: DMA phys=0x{:x}", addr);
            addr
        }
        Err(e) => {
            error!("AHCI: DMA phys failed: {:?}", e);
            loop {
                stem::sleep(Duration::from_secs(60));
            }
        }
    };

    // Layout in physical DMA buffer
    let clb = dma_phys + OFFSET_CMD_LIST as u64;
    let fb = dma_phys + OFFSET_FIS as u64;

    let mut ports: Vec<AhciPort> = Vec::new();

    // Probe ports
    for port_num in 0..32u32 {
        if pi & (1 << port_num) == 0 {
            continue;
        }
        debug!("AHCI: Probing port {}...", port_num);

        let sig = match check_port_type(mapped_base, port_num) {
            Some(s) => s,
            None => {
                debug!("AHCI: Port {} - no device", port_num);
                continue;
            }
        };

        match sig {
            SATA_SIG_ATA => debug!("AHCI: Port {} - SATA drive (sig=0x{:x})", port_num, sig),
            SATA_SIG_ATAPI => {
                debug!("AHCI: Port {} - SATAPI drive (sig=0x{:x})", port_num, sig);

                // Setup Port
                stop_port(mapped_base, port_num);
                let pb = port_base(mapped_base, port_num);
                mmio_write32(pb, PORT_CLB, clb as u32);
                mmio_write32(pb, PORT_CLBU, (clb >> 32) as u32);
                mmio_write32(pb, PORT_FB, fb as u32);
                mmio_write32(pb, PORT_FBU, (fb >> 32) as u32);
                mmio_write32(pb, PORT_IS, 0xFFFFFFFF);
                mmio_write32(pb, PORT_SERR, 0xFFFFFFFF);
                start_port(mapped_base, port_num);

                // Register SATAPI device
                let mut port_info = AhciPort {
                    port_num,
                    sector_count: 0,
                    sector_size: 2048,
                    supports_lba48: false,
                    model: [0u8; 40],
                    serial: [0u8; 20],
                    read_port_handle: None,
                    mmio_base: mapped_base,
                    dma_virt,
                    dma_phys,
                };
                register_atapi_disk(&mut port_info);
                ports.push(port_info);

                stop_port(mapped_base, port_num);
                continue;
            }
            SATA_SIG_SEMB => {
                debug!("AHCI: Port {} - Enclosure (skip)", port_num);
                continue;
            }
            SATA_SIG_PM => {
                debug!("AHCI: Port {} - PM (skip)", port_num);
                continue;
            }
            _ => {
                debug!("AHCI: Port {} - Unknown sig=0x{:x}", port_num, sig);
                continue;
            }
        }

        stop_port(mapped_base, port_num);

        // Setup port command structures
        let pb = port_base(mapped_base, port_num);
        mmio_write32(pb, PORT_CLB, clb as u32);
        mmio_write32(pb, PORT_CLBU, (clb >> 32) as u32);
        mmio_write32(pb, PORT_FB, fb as u32);
        mmio_write32(pb, PORT_FBU, (fb >> 32) as u32);
        mmio_write32(pb, PORT_IS, 0xFFFFFFFF);
        mmio_write32(pb, PORT_SERR, 0xFFFFFFFF);

        start_port(mapped_base, port_num);

        debug!(
            "AHCI: Port {} - SATA drive detected (IDENTIFY deferred - DMA access limitation)",
            port_num
        );

        // Create a placeholder disk entry
        let mut port_info = AhciPort {
            port_num,
            sector_count: 0, // Unknown - would need IDENTIFY
            sector_size: 512,
            supports_lba48: false,
            model: [0u8; 40],
            serial: [0u8; 20],
            read_port_handle: None,
            mmio_base: mapped_base,
            dma_virt,
            dma_phys,
        };
        register_disk(&mut port_info);
        ports.push(port_info);

        stop_port(mapped_base, port_num);
    }

    if ports.is_empty() {
        debug!("AHCI: No SATA disks found");
    } else {
        debug!("AHCI: Found {} SATA disk(s)", ports.len());
    }

    debug!("AHCI: Entering RPC service loop");

    // Collect all port handles for waiting on requests
    let handles: Vec<ChannelHandle> = ports.iter().filter_map(|p| p.read_port_handle).collect();

    if handles.is_empty() {
        info!("AHCI: No active ports to service");
        loop {
            stem::sleep(Duration::from_secs(60));
        }
    }

    // Main service loop
    loop {
        // Wait for a request on any port (blocking)
        let ready_handle = match channel_wait(&handles, abi::syscall::channel_wait::READABLE) {
            Ok(h) => h,
            Err(e) => {
                error!("AHCI: channel_wait failed: {:?}", e);
                stem::sleep(Duration::from_millis(100));
                continue;
            }
        };

        // Find the port that has data
        for port in &mut ports {
            if port.read_port_handle == Some(ready_handle) {
                let mut buf = [0u8; 4096];

                match channel_recv(ready_handle, &mut buf) {
                    Ok(len) if len > 0 => {
                        handle_block_device_request(port, &buf[..len], ready_handle);
                    }
                    Ok(_) => {} // No data
                    Err(e) => {
                        error!("AHCI: channel_recv failed: {:?}", e);
                    }
                }
                break;
            }
        }
    }
}

/// Handle a block device RPC request
fn handle_block_device_request(port: &AhciPort, request_data: &[u8], _service_port: ChannelHandle) {
    if request_data.len() < 5 {
        error!(
            "AHCI: Request too short from client (len={})",
            request_data.len()
        );
        return;
    }

    // Extract response port from header [4: response_port][1: request_type][payload...]
    let response_port = u32::from_le_bytes([
        request_data[0],
        request_data[1],
        request_data[2],
        request_data[3],
    ]) as ChannelHandle;

    let request_type = request_data[4];

    match request_type {
        0 => handle_identify(port, response_port),
        1 => handle_read(port, &request_data[5..], response_port),
        2 => send_error_response(response_port, BlockDeviceError::NotSupported), // Write not supported
        3 => send_error_response(response_port, BlockDeviceError::NotSupported), // Flush not supported
        _ => send_error_response(response_port, BlockDeviceError::InvalidParam),
    }
}

/// Handle Identify request
fn handle_identify(port: &AhciPort, port_handle: ChannelHandle) {
    let response = IdentifyResponse {
        sector_size: port.sector_size,
        sector_count: port.sector_count,
        model: port.model,
        serial: port.serial,
        flags: if port.supports_lba48 {
            device_flags::LBA48
        } else {
            0
        },
    };

    let mut response_buf = [0u8; core::mem::size_of::<IdentifyResponse>() + 1];
    response_buf[0] = BlockDeviceResponse::Ok as u8;

    unsafe {
        core::ptr::copy_nonoverlapping(
            &response as *const _ as *const u8,
            response_buf[1..].as_mut_ptr(),
            core::mem::size_of::<IdentifyResponse>(),
        );
    }

    if let Err(e) = channel_send(port_handle, &response_buf) {
        error!("AHCI: Failed to send Identify response: {:?}", e);
    }
}

/// Handle Read request
fn handle_read(port: &AhciPort, request_data: &[u8], port_handle: ChannelHandle) {
    if request_data.len() < core::mem::size_of::<ReadRequest>() {
        send_error_response(port_handle, BlockDeviceError::InvalidParam);
        return;
    }

    let req: ReadRequest =
        unsafe { core::ptr::read_unaligned(request_data.as_ptr() as *const ReadRequest) };

    // Validate sector count based on sector size to ensure response fits in port buffer
    let max_sectors = if port.sector_size == 2048 {
        MAX_SECTORS_PER_READ_2048
    } else {
        MAX_SECTORS_PER_READ_512
    };

    if req.sector_count == 0 || req.sector_count > max_sectors {
        send_error_response(port_handle, BlockDeviceError::InvalidParam);
        return;
    }

    if port.sector_count > 0 && req.lba.saturating_add(req.sector_count as u64) > port.sector_count
    {
        send_error_response(port_handle, BlockDeviceError::OutOfRange);
        return;
    }

    // Create block device wrapper for reading
    if port.sector_size == 2048 {
        // ATAPI device
        let atapi_dev = AhciAtapiDevice {
            mmio_base: port.mmio_base,
            port: port.port_num,
            dma_virt: port.dma_virt,
            dma_phys: port.dma_phys,
        };

        // Read the data
        let bytes_to_read = (req.sector_count as usize) * 2048;
        let mut data = Vec::with_capacity(bytes_to_read);
        data.resize(bytes_to_read, 0);

        match atapi_dev.read_sectors(req.lba, req.sector_count as u64, &mut data) {
            Ok(_) => send_read_response(port_handle, &data),
            Err(_) => send_error_response(port_handle, BlockDeviceError::IoError),
        }
    } else {
        // ATA device - not fully implemented due to DMA limitations
        // For now, return error
        send_error_response(port_handle, BlockDeviceError::NotSupported);
    }
}

/// Send a Read success response
fn send_read_response(port_handle: ChannelHandle, data: &[u8]) {
    let header = ReadResponse {
        data_len: data.len() as u32,
    };

    let response_size = 1 + core::mem::size_of::<ReadResponse>() + data.len();
    let mut response_buf = Vec::with_capacity(response_size);
    response_buf.push(BlockDeviceResponse::Ok as u8);

    unsafe {
        let header_bytes = core::slice::from_raw_parts(
            &header as *const _ as *const u8,
            core::mem::size_of::<ReadResponse>(),
        );
        response_buf.extend_from_slice(header_bytes);
    }

    response_buf.extend_from_slice(data);

    if let Err(e) = channel_send(port_handle, &response_buf) {
        error!("AHCI: Failed to send Read response: {:?}", e);
    }
}

/// Send an error response
fn send_error_response(port_handle: ChannelHandle, error_code: BlockDeviceError) {
    let error_resp = ErrorResponse {
        error_code: error_code as u8,
        _reserved: [0; 3],
    };

    let mut response_buf = [0u8; 1 + core::mem::size_of::<ErrorResponse>()];
    response_buf[0] = BlockDeviceResponse::Error as u8;

    unsafe {
        core::ptr::copy_nonoverlapping(
            &error_resp as *const _ as *const u8,
            response_buf[1..].as_mut_ptr(),
            core::mem::size_of::<ErrorResponse>(),
        );
    }

    if let Err(e) = channel_send(port_handle, &response_buf) {
        error!("AHCI: Failed to send Error response: {:?}", e);
    }
}

/// Scan `/sys/devices` for an AHCI controller (PCI class 0x010601) and return its sysfs path.
fn find_ahci_device() -> Option<alloc::string::String> {
    use abi::syscall::vfs_flags;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_readdir};

    let fd = match vfs_open("/sys/devices", vfs_flags::O_RDONLY) {
        Ok(fd) => fd,
        Err(_) => return None,
    };

    let mut buf = [0u8; 4096];
    let n = match vfs_readdir(fd, &mut buf) {
        Ok(n) => n,
        Err(_) => {
            let _ = vfs_close(fd);
            return None;
        }
    };
    let _ = vfs_close(fd);

    let mut pos = 0;
    while pos < n {
        let entry_buf = &buf[pos..n];
        let name = core::str::from_utf8(entry_buf)
            .unwrap_or("")
            .split('\0')
            .next()
            .unwrap_or("");
        if name.is_empty() {
            break;
        }

        if name.starts_with("pci-") {
            let class_path = alloc::format!("/sys/devices/{}/class", name);
            if let Some(class_str) = read_sys_string(&class_path) {
                // PCI Class 01, Subclass 06, ProgIf 01 is AHCI
                if class_str.trim().starts_with("0x010601") {
                    let dev_path = alloc::format!("/sys/devices/{}", name);
                    info!("AHCI: Found {} via scan", dev_path);
                    return Some(dev_path);
                }
            }
        }
        pos += name.len() + 1;
    }
    None
}

fn read_sys_string(path: &str) -> Option<alloc::string::String> {
    use abi::syscall::vfs_flags::O_RDONLY;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read};

    let fd = vfs_open(path, O_RDONLY).ok()?;
    let mut buf = [0u8; 128];
    let n = vfs_read(fd, &mut buf).ok()?;
    let _ = vfs_close(fd);

    Some(
        alloc::string::String::from_utf8_lossy(&buf[..n])
            .trim()
            .to_string(),
    )
}
