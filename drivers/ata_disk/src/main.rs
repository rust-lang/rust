//! ATA Disk Driver (Read-only v0)
//!
//! Userspace driver that detects ATA devices on legacy ports and registers
//! them in the System Graph. Uses ioport_read/write syscalls for PIO access.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use alloc::vec;
use alloc::vec::Vec;
use core::time::Duration;
use stem::abi::block_device_protocol::*;
use stem::abi::module_manifest::{ManifestHeader, ModuleKind, MANIFEST_MAGIC};
use stem::syscall::{channel_create, channel_recv, channel_send, channel_wait, ChannelHandle};
use stem::syscall::{ioport_read, ioport_write};
use stem::{error, info};

#[unsafe(link_section = ".thing_manifest")]
#[unsafe(no_mangle)]
#[used]
pub static MANIFEST: ManifestHeader = ManifestHeader {
    magic: MANIFEST_MAGIC,
    kind: ModuleKind::Service,
    device_kind: *b"dev.storage.ata\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
    version: 1,
    _reserved: 0,
};

// ATA port bases (legacy)
const ATA_PRIMARY_IO: u16 = 0x1F0;
const ATA_PRIMARY_CTRL: u16 = 0x3F6;
const ATA_SECONDARY_IO: u16 = 0x170;
const ATA_SECONDARY_CTRL: u16 = 0x376;

// ATA register offsets from IO base
const ATA_REG_DATA: u16 = 0;
const ATA_REG_SECCOUNT: u16 = 2;
const ATA_REG_LBA_LO: u16 = 3;
const ATA_REG_LBA_MID: u16 = 4;
const ATA_REG_LBA_HI: u16 = 5;
const ATA_REG_DRIVE: u16 = 6;
const ATA_REG_STATUS: u16 = 7;
const ATA_REG_COMMAND: u16 = 7;

// ATA commands
const ATA_CMD_IDENTIFY: u8 = 0xEC;
const ATA_CMD_IDENTIFY_PACKET: u8 = 0xA1;
const ATA_CMD_PACKET: u8 = 0xA0;
const ATA_CMD_READ_SECTORS: u8 = 0x20; // LBA28
const ATA_CMD_READ_SECTORS_EXT: u8 = 0x24; // LBA48

// ATAPI signatures (after IDENTIFY, in LBA mid/hi)
const ATAPI_SIG_MID: u8 = 0x14;
const ATAPI_SIG_HI: u8 = 0xEB;

// ATAPI CD-ROM sector size
const ATAPI_SECTOR_SIZE: u64 = 2048;

// Status bits
const ATA_SR_BSY: u8 = 0x80;
const ATA_SR_DRQ: u8 = 0x08;
const ATA_SR_ERR: u8 = 0x01;

struct AtaDisk {
    io_base: u16,
    is_slave: bool,
    sector_count: u64,
    sector_size: u32,
    supports_lba48: bool,
    model: [u8; 40],
    serial: [u8; 20],
    read_port_handle: Option<ChannelHandle>,
}

/// ATAPI (CD-ROM) device
struct AtapiDevice {
    io_base: u16,
    ctrl_base: u16,
    is_slave: bool,
    sector_size: u32,
    sector_count: u64,
    model: [u8; 40],
    serial: [u8; 20],
    read_port_handle: Option<ChannelHandle>,
}

fn ata_inb(port: u16) -> u8 {
    ioport_read(port as usize, 1) as u8
}

fn ata_inw(port: u16) -> u16 {
    ioport_read(port as usize, 2) as u16
}

fn ata_outb(port: u16, val: u8) {
    ioport_write(port as usize, val as usize, 1);
}

fn wait_bsy_clear(io_base: u16) -> bool {
    for _ in 0..100000 {
        let status = ata_inb(io_base + ATA_REG_STATUS);
        if status & ATA_SR_BSY == 0 {
            return true;
        }
    }
    false
}

fn identify_drive(io_base: u16, ctrl_base: u16, is_slave: bool) -> Option<AtaDisk> {
    // Select drive
    let drive_sel = if is_slave { 0xB0 } else { 0xA0 };
    ata_outb(io_base + ATA_REG_DRIVE, drive_sel);

    // Small delay (read alternate status 4 times)
    for _ in 0..4 {
        ata_inb(ctrl_base);
    }

    // Clear sector count and LBA registers
    ata_outb(io_base + ATA_REG_SECCOUNT, 0);
    ata_outb(io_base + ATA_REG_LBA_LO, 0);
    ata_outb(io_base + ATA_REG_LBA_MID, 0);
    ata_outb(io_base + ATA_REG_LBA_HI, 0);

    // Send IDENTIFY command
    ata_outb(io_base + ATA_REG_COMMAND, ATA_CMD_IDENTIFY);

    // Check if drive exists
    let status = ata_inb(io_base + ATA_REG_STATUS);
    if status == 0 || status == 0xFF {
        return None; // No drive
    }

    // Wait for BSY to clear
    if !wait_bsy_clear(io_base) {
        return None;
    }

    // Check for ATAPI (different signature in LBA mid/hi)
    let lba_mid = ata_inb(io_base + ATA_REG_LBA_MID);
    let lba_hi = ata_inb(io_base + ATA_REG_LBA_HI);
    if lba_mid == ATAPI_SIG_MID && lba_hi == ATAPI_SIG_HI {
        return None; // ATAPI device - handled separately
    }
    if lba_mid != 0 || lba_hi != 0 {
        return None; // Unknown signature
    }

    // Wait for DRQ or ERR
    loop {
        let status = ata_inb(io_base + ATA_REG_STATUS);
        if status & ATA_SR_DRQ != 0 {
            break;
        }
        if status & ATA_SR_ERR != 0 {
            return None;
        }
        if status == 0 {
            return None;
        }
    }

    // Read 256 words of identification data
    let mut ident = [0u16; 256];
    for i in 0..256 {
        ident[i] = ata_inw(io_base + ATA_REG_DATA);
    }

    // Parse identification data
    let supports_lba48 = (ident[83] & (1 << 10)) != 0;

    let sector_count = if supports_lba48 {
        (ident[100] as u64)
            | ((ident[101] as u64) << 16)
            | ((ident[102] as u64) << 32)
            | ((ident[103] as u64) << 48)
    } else {
        (ident[60] as u64) | ((ident[61] as u64) << 16)
    };

    // Extract model string (words 27-46, byte-swapped)
    let mut model = [0u8; 40];
    for i in 0..20 {
        let word = ident[27 + i];
        model[i * 2] = (word >> 8) as u8;
        model[i * 2 + 1] = (word & 0xFF) as u8;
    }

    Some(AtaDisk {
        io_base,
        is_slave,
        sector_count,
        sector_size: 512,
        supports_lba48,
        model,
        serial: [0u8; 20],
        read_port_handle: None,
    })
}

fn read_sectors(
    disk: &AtaDisk,
    lba: u64,
    count: u16,
    buf: &mut [u8],
) -> Result<usize, &'static str> {
    if count == 0 || count > 256 {
        return Err("Invalid sector count");
    }

    let bytes_needed = count as usize * 512;
    if buf.len() < bytes_needed {
        return Err("Buffer too small");
    }

    if !disk.supports_lba48 && lba > 0x0FFFFFFF {
        return Err("LBA out of range for LBA28");
    }

    let drive_sel = if disk.is_slave { 0xF0 } else { 0xE0 };

    if disk.supports_lba48 {
        ata_outb(disk.io_base + ATA_REG_DRIVE, drive_sel);
        ata_outb(disk.io_base + ATA_REG_SECCOUNT, ((count >> 8) & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_LO, ((lba >> 24) & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_MID, ((lba >> 32) & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_HI, ((lba >> 40) & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_SECCOUNT, (count & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_LO, (lba & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_MID, ((lba >> 8) & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_HI, ((lba >> 16) & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_COMMAND, ATA_CMD_READ_SECTORS_EXT);
    } else {
        let lba28 = lba as u32;
        ata_outb(
            disk.io_base + ATA_REG_DRIVE,
            drive_sel | ((lba28 >> 24) & 0x0F) as u8,
        );
        ata_outb(disk.io_base + ATA_REG_SECCOUNT, count as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_LO, (lba28 & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_MID, ((lba28 >> 8) & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_LBA_HI, ((lba28 >> 16) & 0xFF) as u8);
        ata_outb(disk.io_base + ATA_REG_COMMAND, ATA_CMD_READ_SECTORS);
    }

    let mut offset = 0;
    for _ in 0..count {
        loop {
            let status = ata_inb(disk.io_base + ATA_REG_STATUS);
            if status & ATA_SR_ERR != 0 {
                return Err("Read error");
            }
            if status & ATA_SR_DRQ != 0 {
                break;
            }
        }

        for _ in 0..256 {
            let word = ata_inw(disk.io_base + ATA_REG_DATA);
            buf[offset] = (word & 0xFF) as u8;
            buf[offset + 1] = (word >> 8) as u8;
            offset += 2;
        }
    }

    Ok(bytes_needed)
}

fn register_disk(disk: &mut AtaDisk, channel: &str, drive: &str) {
    // Create RPC port for block device service (4KB buffer)
    let (write_handle, read_handle) = match channel_create(4096) {
        Ok(handles) => handles,
        Err(e) => {
            error!("ATA_DISK: Failed to create port: {:?}", e);
            return;
        }
    };
    disk.read_port_handle = Some(read_handle);

    // Publish to VFS
    use stem::syscall::vfs::{vfs_close, vfs_mkdir, vfs_open, vfs_write};
    let _ = vfs_mkdir("/services/storage");
    let name = alloc::format!("/services/storage/ata_{}_{}", channel, drive);
    if let Ok(fd) = vfs_open(
        &name,
        abi::syscall::vfs_flags::O_CREAT | abi::syscall::vfs_flags::O_RDWR,
    ) {
        let _ = vfs_write(fd, alloc::format!("{}", write_handle).as_bytes());
        let _ = vfs_close(fd);
    }

    let model_str = core::str::from_utf8(&disk.model)
        .unwrap_or("Unknown")
        .trim();

    info!(
        "ATA_DISK: Registered disk ch={} drv={} sectors={} lba48={} model='{}' rpc_port={}",
        channel, drive, disk.sector_count, disk.supports_lba48, model_str, write_handle
    );
}

/// Identify an ATAPI device (CD-ROM, DVD, etc.)
fn identify_atapi(io_base: u16, ctrl_base: u16, is_slave: bool) -> Option<AtapiDevice> {
    // Select drive
    let drive_sel = if is_slave { 0xB0 } else { 0xA0 };
    ata_outb(io_base + ATA_REG_DRIVE, drive_sel);

    // Small delay
    for _ in 0..4 {
        ata_inb(ctrl_base);
    }

    // Clear registers
    ata_outb(io_base + ATA_REG_SECCOUNT, 0);
    ata_outb(io_base + ATA_REG_LBA_LO, 0);
    ata_outb(io_base + ATA_REG_LBA_MID, 0);
    ata_outb(io_base + ATA_REG_LBA_HI, 0);

    // Send IDENTIFY command first to detect signature
    ata_outb(io_base + ATA_REG_COMMAND, ATA_CMD_IDENTIFY);

    let status = ata_inb(io_base + ATA_REG_STATUS);
    if status == 0 || status == 0xFF {
        return None;
    }

    if !wait_bsy_clear(io_base) {
        return None;
    }

    // Check for ATAPI signature
    let lba_mid = ata_inb(io_base + ATA_REG_LBA_MID);
    let lba_hi = ata_inb(io_base + ATA_REG_LBA_HI);
    if lba_mid != ATAPI_SIG_MID || lba_hi != ATAPI_SIG_HI {
        return None;
    }

    // Now send IDENTIFY PACKET DEVICE
    ata_outb(io_base + ATA_REG_COMMAND, ATA_CMD_IDENTIFY_PACKET);

    if !wait_bsy_clear(io_base) {
        return None;
    }

    // Wait for DRQ
    for _ in 0..10000 {
        let status = ata_inb(io_base + ATA_REG_STATUS);
        if status & ATA_SR_ERR != 0 {
            return None;
        }
        if status & ATA_SR_DRQ != 0 {
            break;
        }
    }

    // Read 256 words of identification
    let mut ident = [0u16; 256];
    for i in 0..256 {
        ident[i] = ata_inw(io_base + ATA_REG_DATA);
    }

    // Extract model (words 27-46, byte-swapped)
    let mut model = [0u8; 40];
    for i in 0..20 {
        let word = ident[27 + i];
        model[i * 2] = (word >> 8) as u8;
        model[i * 2 + 1] = (word & 0xFF) as u8;
    }

    Some(AtapiDevice {
        io_base,
        ctrl_base,
        is_slave,
        sector_size: 2048,
        sector_count: 0,
        model,
        serial: [0u8; 20],
        read_port_handle: None,
    })
}

/// Read sectors from ATAPI device using SCSI READ(12) packet command.
/// Sector size is 2048 bytes for CD-ROM.
pub fn atapi_read_sectors(
    dev: &AtapiDevice,
    lba: u64,
    count: u32,
    buf: &mut [u8],
) -> Result<usize, &'static str> {
    if count == 0 || count > 32 {
        return Err("Invalid sector count");
    }
    let bytes_needed = count as usize * ATAPI_SECTOR_SIZE as usize;
    if buf.len() < bytes_needed {
        return Err("Buffer too small");
    }

    // Select drive
    let drive_sel = if dev.is_slave { 0xB0 } else { 0xA0 };
    ata_outb(dev.io_base + ATA_REG_DRIVE, drive_sel);

    // Delay
    for _ in 0..4 {
        ata_inb(dev.ctrl_base);
    }

    if !wait_bsy_clear(dev.io_base) {
        return Err("Device busy");
    }

    // Set byte count limit (max transfer size)
    let byte_count = bytes_needed as u16;
    ata_outb(dev.io_base + ATA_REG_LBA_MID, (byte_count & 0xFF) as u8);
    ata_outb(
        dev.io_base + ATA_REG_LBA_HI,
        ((byte_count >> 8) & 0xFF) as u8,
    );

    // Send PACKET command
    ata_outb(dev.io_base + ATA_REG_COMMAND, ATA_CMD_PACKET);

    // Wait for DRQ (ready to receive packet)
    for _ in 0..100000 {
        let status = ata_inb(dev.io_base + ATA_REG_STATUS);
        if status & ATA_SR_ERR != 0 {
            return Err("Packet command error");
        }
        if status & ATA_SR_DRQ != 0 {
            break;
        }
    }

    // Build SCSI READ(12) command (12 bytes, padded to 6 words)
    let lba32 = lba as u32;
    let packet: [u16; 6] = [
        0x00A8, // READ(12) opcode = 0xA8, flags = 0
        ((lba32 >> 24) as u16) << 8 | ((lba32 >> 16) as u16 & 0xFF), // LBA high
        ((lba32 >> 8) as u16 & 0xFF) << 8 | (lba32 as u16 & 0xFF), // LBA low
        ((count >> 24) as u16) << 8 | ((count >> 16) as u16 & 0xFF), // Transfer length high
        ((count >> 8) as u16 & 0xFF) << 8 | (count as u16 & 0xFF), // Transfer length low
        0x0000, // Control
    ];

    // Send packet (6 words)
    for word in packet {
        ata_outw(dev.io_base + ATA_REG_DATA, word);
    }

    // Read data
    let mut offset = 0;
    for _ in 0..count {
        // Wait for DRQ
        loop {
            let status = ata_inb(dev.io_base + ATA_REG_STATUS);
            if status & ATA_SR_ERR != 0 {
                return Err("Read error");
            }
            if status & ATA_SR_BSY == 0 && status & ATA_SR_DRQ != 0 {
                break;
            }
        }

        // Read 2048 bytes (1024 words)
        for _ in 0..1024 {
            let word = ata_inw(dev.io_base + ATA_REG_DATA);
            buf[offset] = (word & 0xFF) as u8;
            buf[offset + 1] = (word >> 8) as u8;
            offset += 2;
        }
    }

    Ok(bytes_needed)
}

fn ata_outw(port: u16, val: u16) {
    // Write 16-bit word using ioport_write with size=2
    ioport_write(port as usize, val as usize, 2);
}

fn register_atapi(dev: &mut AtapiDevice, channel: &str, drive: &str) {
    // Create RPC port for block device service (4KB buffer)
    let (write_handle, read_handle) = match channel_create(4096) {
        Ok(handles) => handles,
        Err(e) => {
            error!("ATA_DISK: Failed to create port: {:?}", e);
            return;
        }
    };
    dev.read_port_handle = Some(read_handle);

    // Publish to VFS
    use stem::syscall::vfs::{vfs_close, vfs_mkdir, vfs_open, vfs_write};
    let _ = vfs_mkdir("/services/storage");
    let name = alloc::format!("/services/storage/atapi_{}_{}", channel, drive);
    if let Ok(fd) = vfs_open(
        &name,
        abi::syscall::vfs_flags::O_CREAT | abi::syscall::vfs_flags::O_RDWR,
    ) {
        let _ = vfs_write(fd, alloc::format!("{}", write_handle).as_bytes());
        let _ = vfs_close(fd);
    }

    let model_str = core::str::from_utf8(&dev.model)
        .unwrap_or("ATAPI Device")
        .trim();

    info!(
        "ATA_DISK: Registered ATAPI ch={} drv={} model='{}' rpc_port={}",
        channel, drive, model_str, write_handle
    );
}

#[stem::main]
fn main(_arg: usize) -> ! {
    info!("ATA_DISK: Starting ATA disk driver v1 (with ATAPI support)");

    let mut disks: Vec<AtaDisk> = Vec::new();
    let mut atapi_devs: Vec<AtapiDevice> = Vec::new();

    // Probe primary channel - ATA
    info!("ATA_DISK: Probing primary channel (0x1F0)...");
    if let Some(mut disk) = identify_drive(ATA_PRIMARY_IO, ATA_PRIMARY_CTRL, false) {
        info!("ATA_DISK: Found primary master (ATA)");
        register_disk(&mut disk, "primary", "master");
        disks.push(disk);
    } else if let Some(mut dev) = identify_atapi(ATA_PRIMARY_IO, ATA_PRIMARY_CTRL, false) {
        info!("ATA_DISK: Found primary master (ATAPI)");
        register_atapi(&mut dev, "primary", "master");
        atapi_devs.push(dev);
    }

    if let Some(mut disk) = identify_drive(ATA_PRIMARY_IO, ATA_PRIMARY_CTRL, true) {
        info!("ATA_DISK: Found primary slave (ATA)");
        register_disk(&mut disk, "primary", "slave");
        disks.push(disk);
    } else if let Some(mut dev) = identify_atapi(ATA_PRIMARY_IO, ATA_PRIMARY_CTRL, true) {
        info!("ATA_DISK: Found primary slave (ATAPI)");
        register_atapi(&mut dev, "primary", "slave");
        atapi_devs.push(dev);
    }

    // Probe secondary channel
    info!("ATA_DISK: Probing secondary channel (0x170)...");
    if let Some(mut disk) = identify_drive(ATA_SECONDARY_IO, ATA_SECONDARY_CTRL, false) {
        info!("ATA_DISK: Found secondary master (ATA)");
        register_disk(&mut disk, "secondary", "master");
        disks.push(disk);
    } else if let Some(mut dev) = identify_atapi(ATA_SECONDARY_IO, ATA_SECONDARY_CTRL, false) {
        info!("ATA_DISK: Found secondary master (ATAPI)");
        register_atapi(&mut dev, "secondary", "master");
        atapi_devs.push(dev);
    }

    if let Some(mut disk) = identify_drive(ATA_SECONDARY_IO, ATA_SECONDARY_CTRL, true) {
        info!("ATA_DISK: Found secondary slave (ATA)");
        register_disk(&mut disk, "secondary", "slave");
        disks.push(disk);
    } else if let Some(mut dev) = identify_atapi(ATA_SECONDARY_IO, ATA_SECONDARY_CTRL, true) {
        info!("ATA_DISK: Found secondary slave (ATAPI)");
        register_atapi(&mut dev, "secondary", "slave");
        atapi_devs.push(dev);
    }

    // Summary
    info!(
        "ATA_DISK: Found {} ATA disk(s), {} ATAPI device(s)",
        disks.len(),
        atapi_devs.len()
    );

    info!("ATA_DISK: Entering RPC service loop");

    // Collect all port handles for waiting on requests
    let mut handles: Vec<ChannelHandle> = Vec::new();
    for disk in &disks {
        if let Some(h) = disk.read_port_handle {
            handles.push(h);
        }
    }
    for dev in &atapi_devs {
        if let Some(h) = dev.read_port_handle {
            handles.push(h);
        }
    }

    if handles.is_empty() {
        info!("ATA_DISK: No active devices to service");
        loop {
            stem::syscall::sleep_ms(60_000);
        }
    }

    // Main service loop
    loop {
        // Wait for a request on any port (blocking)
        let ready_handle = match channel_wait(&handles, abi::syscall::channel_wait::READABLE) {
            Ok(h) => h,
            Err(e) => {
                error!("ATA_DISK: channel_wait failed: {:?}", e);
                stem::sleep(Duration::from_millis(100));
                continue;
            }
        };

        // Find the device that has data
        let mut found = false;
        for disk in &disks {
            if disk.read_port_handle == Some(ready_handle) {
                let mut buf = [0u8; 4096];
                match channel_recv(ready_handle, &mut buf) {
                    Ok(len) if len > 0 => {
                        handle_ata_request(disk, &buf[..len], ready_handle);
                    }
                    Ok(_) => {} // No data
                    Err(e) => {
                        error!("ATA_DISK: channel_recv failed: {:?}", e);
                    }
                }
                found = true;
                break;
            }
        }

        if !found {
            for dev in &atapi_devs {
                if dev.read_port_handle == Some(ready_handle) {
                    let mut buf = [0u8; 4096];
                    match channel_recv(ready_handle, &mut buf) {
                        Ok(len) if len > 0 => {
                            handle_atapi_request(dev, &buf[..len], ready_handle);
                        }
                        Ok(_) => {} // No data
                        Err(e) => {
                            error!("ATA_DISK: channel_recv failed: {:?}", e);
                        }
                    }
                    break;
                }
            }
        }
    }
}

/// Handle a block device RPC request for ATA disk
fn handle_ata_request(disk: &AtaDisk, request_data: &[u8], port_handle: ChannelHandle) {
    if request_data.is_empty() {
        send_error_response(port_handle, BlockDeviceError::InvalidParam);
        return;
    }

    let request_type = request_data[0];

    match request_type {
        0 => handle_ata_identify(disk, port_handle),
        1 => handle_ata_read(disk, &request_data[1..], port_handle),
        2 => send_error_response(port_handle, BlockDeviceError::NotSupported), // Write not supported
        3 => send_error_response(port_handle, BlockDeviceError::NotSupported), // Flush not supported
        _ => send_error_response(port_handle, BlockDeviceError::InvalidParam),
    }
}

/// Handle Identify request for ATA disk
fn handle_ata_identify(disk: &AtaDisk, port_handle: ChannelHandle) {
    let response = IdentifyResponse {
        sector_size: disk.sector_size,
        sector_count: disk.sector_count,
        model: disk.model,
        serial: disk.serial,
        flags: if disk.supports_lba48 {
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
        error!("ATA_DISK: Failed to send Identify response: {:?}", e);
    }
}

/// Handle Read request for ATA disk
fn handle_ata_read(disk: &AtaDisk, request_data: &[u8], port_handle: ChannelHandle) {
    if request_data.len() < core::mem::size_of::<ReadRequest>() {
        send_error_response(port_handle, BlockDeviceError::InvalidParam);
        return;
    }

    let req: ReadRequest =
        unsafe { core::ptr::read_unaligned(request_data.as_ptr() as *const ReadRequest) };

    // Validate sector count (max 7 sectors of 512 bytes to fit in 4KB port buffer)
    if req.sector_count == 0 || req.sector_count > 7 {
        send_error_response(port_handle, BlockDeviceError::InvalidParam);
        return;
    }

    if disk.sector_count > 0 && req.lba.saturating_add(req.sector_count as u64) > disk.sector_count
    {
        send_error_response(port_handle, BlockDeviceError::OutOfRange);
        return;
    }

    // Read the data
    let bytes_to_read = (req.sector_count as usize) * 512;
    let mut data = vec![0u8; bytes_to_read];

    match read_sectors(disk, req.lba, req.sector_count as u16, &mut data) {
        Ok(_) => send_read_response(port_handle, &data),
        Err(_) => send_error_response(port_handle, BlockDeviceError::IoError),
    }
}

/// Handle a block device RPC request for ATAPI device
fn handle_atapi_request(dev: &AtapiDevice, request_data: &[u8], port_handle: ChannelHandle) {
    if request_data.is_empty() {
        send_error_response(port_handle, BlockDeviceError::InvalidParam);
        return;
    }

    let request_type = request_data[0];

    match request_type {
        0 => handle_atapi_identify(dev, port_handle),
        1 => handle_atapi_read(dev, &request_data[1..], port_handle),
        2 => send_error_response(port_handle, BlockDeviceError::NotSupported), // Write not supported
        3 => send_error_response(port_handle, BlockDeviceError::NotSupported), // Flush not supported
        _ => send_error_response(port_handle, BlockDeviceError::InvalidParam),
    }
}

/// Handle Identify request for ATAPI device
fn handle_atapi_identify(dev: &AtapiDevice, port_handle: ChannelHandle) {
    let response = IdentifyResponse {
        sector_size: dev.sector_size,
        sector_count: dev.sector_count,
        model: dev.model,
        serial: dev.serial,
        flags: 0,
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
        error!("ATA_DISK: Failed to send Identify response: {:?}", e);
    }
}

/// Handle Read request for ATAPI device
fn handle_atapi_read(dev: &AtapiDevice, request_data: &[u8], port_handle: ChannelHandle) {
    if request_data.len() < core::mem::size_of::<ReadRequest>() {
        send_error_response(port_handle, BlockDeviceError::InvalidParam);
        return;
    }

    let req: ReadRequest =
        unsafe { core::ptr::read_unaligned(request_data.as_ptr() as *const ReadRequest) };

    // Validate sector count (max 1 sector of 2048 bytes to fit in 4KB port buffer)
    if req.sector_count == 0 || req.sector_count > 1 {
        send_error_response(port_handle, BlockDeviceError::InvalidParam);
        return;
    }

    if dev.sector_count > 0 && req.lba.saturating_add(req.sector_count as u64) > dev.sector_count {
        send_error_response(port_handle, BlockDeviceError::OutOfRange);
        return;
    }

    // Read the data
    let bytes_to_read = (req.sector_count as usize) * 2048;
    let mut data = vec![0u8; bytes_to_read];

    match atapi_read_sectors(dev, req.lba, req.sector_count, &mut data) {
        Ok(_) => send_read_response(port_handle, &data),
        Err(_) => send_error_response(port_handle, BlockDeviceError::IoError),
    }
}

/// Send a Read success response
fn send_read_response(port_handle: ChannelHandle, data: &[u8]) {
    let header = ReadResponse {
        data_len: data.len() as u32,
    };

    let response_size = 1 + core::mem::size_of::<ReadResponse>() + data.len();
    let mut response_buf = vec![0u8; response_size];
    response_buf[0] = BlockDeviceResponse::Ok as u8;

    unsafe {
        let header_bytes = core::slice::from_raw_parts(
            &header as *const _ as *const u8,
            core::mem::size_of::<ReadResponse>(),
        );
        response_buf[1..1 + core::mem::size_of::<ReadResponse>()].copy_from_slice(header_bytes);
    }

    response_buf[1 + core::mem::size_of::<ReadResponse>()..].copy_from_slice(data);

    if let Err(e) = channel_send(port_handle, &response_buf) {
        error!("ATA_DISK: Failed to send Read response: {:?}", e);
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
        error!("ATA_DISK: Failed to send Error response: {:?}", e);
    }
}
