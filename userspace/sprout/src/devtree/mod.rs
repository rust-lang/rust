#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
pub mod aarch64;
pub mod loongarch64;
pub mod riscv64;
pub mod x86_64;

use abi::ids::HandleId;
use abi::schema::{confidence, keys, source};
use alloc::vec;
use stem::{debug, info};

pub fn set_str_prop(_id: abi::types::ThingId, _key: &str, _val: &str) -> Result<(), ()> {
    Ok(())
}

#[allow(dead_code)]
pub struct DevTreeCtx {
    pub host: abi::types::ThingId,
    pub platform_bus: abi::types::ThingId,
    pub hhdm: usize,
    pub acpi_rsdp: Option<usize>,
    pub dtb_ptr: Option<usize>,
    pub dtb_bytespace: Option<abi::types::ThingId>,
    pub dtb_node_id: Option<abi::types::ThingId>,
}

pub fn init() -> Result<DevTreeCtx, ()> {
    debug!("SPROUT: devtree::init entry (VFS-native)");

    // 1. Get HHDM Offset
    debug!("SPROUT: Reading HHDM offset from /sys/firmware/hhdm");
    let hhdm = read_sys_u64("/sys/firmware/hhdm").unwrap_or(0) as usize;
    if hhdm == 0 {
        stem::warn!("SPROUT: Failed to read HHDM offset!");
    }

    // 2. Check for Firmware
    let mut acpi_rsdp = None;
    let mut dtb_ptr = None;

    if let Ok(val) = read_sys_u64("/sys/firmware/acpi") {
        acpi_rsdp = Some(val as usize);
        debug!("SPROUT: ACPI RSDP = 0x{:x}", val);
    }

    if let Ok(val) = read_sys_u64("/sys/firmware/dtb") {
        dtb_ptr = Some(val as usize);
        debug!("SPROUT: DTB PHYS = 0x{:x}", val);
    }

    debug!("SPROUT: Init OK, returning context (graph discovery eradicated)");
    Ok(DevTreeCtx {
        host: abi::types::ThingId::default(),
        platform_bus: abi::types::ThingId::default(),
        hhdm,
        acpi_rsdp,
        dtb_ptr,
        dtb_bytespace: None,
        dtb_node_id: None,
    })
}

fn read_sys_u64(path: &str) -> Result<u64, ()> {
    use abi::syscall::vfs_flags::O_RDONLY;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read};

    let fd = vfs_open(path, O_RDONLY).map_err(|_| ())?;
    let mut buf = [0u8; 32];
    let n = vfs_read(fd, &mut buf).map_err(|_| ())?;
    let _ = vfs_close(fd);

    let s = core::str::from_utf8(&buf[..n]).map_err(|_| ())?;
    let trimmed = s.trim();
    if trimmed.starts_with("0x") {
        u64::from_str_radix(&trimmed[2..], 16).map_err(|_| ())
    } else {
        trimmed.parse::<u64>().map_err(|_| ())
    }
}

pub fn build(_ctx: &DevTreeCtx) -> Result<(), ()> {
    debug!("SPROUT: build() called (VFS-native, graph building skipped)");
    Ok(())
}
