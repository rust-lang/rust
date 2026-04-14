#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use abi::errors::Errno;
use abi::syscall::vfs_flags::O_RDONLY;
use alloc::string::String;
use alloc::vec::Vec;
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read, vfs_readdir};

#[derive(Clone)]
pub struct SysDevice {
    pub slot: String,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u32, // Full class triplet: class << 16 | subclass << 8 | prog_if
    pub present: bool,
}

pub fn scan_devices() -> Result<Vec<SysDevice>, Errno> {
    let slots = read_dir("/sys/devices")?;
    let mut devices = Vec::new();
    for slot in slots {
        if !slot.starts_with("pci-") {
            continue;
        }

        let base = alloc::format!("/sys/devices/{}", slot);
        let vendor_id = read_hex_u16(&alloc::format!("{}/vendor", base))?;
        let device_id = read_hex_u16(&alloc::format!("{}/device", base))?;
        let class_triplet = read_hex_u32(&alloc::format!("{}/class", base))?;
        let status = read_string(&alloc::format!("{}/status", base))?;

        devices.push(SysDevice {
            slot,
            vendor_id,
            device_id,
            class_code: class_triplet,
            present: status == "present",
        });
    }
    Ok(devices)
}

pub fn device_present(slot: &str) -> bool {
    let path = alloc::format!("/sys/devices/{}/status", slot);
    matches!(read_string(&path), Ok(status) if status == "present")
}

fn read_dir(path: &str) -> Result<Vec<String>, Errno> {
    let fd = vfs_open(path, O_RDONLY)?;
    let mut buf = [0u8; 4096];
    let n = vfs_readdir(fd, &mut buf)?;
    let _ = vfs_close(fd);

    let mut entries = Vec::new();
    let mut start = 0usize;
    while start < n {
        let mut end = start;
        while end < n && buf[end] != 0 {
            end += 1;
        }
        if end > start {
            if let Ok(name) = core::str::from_utf8(&buf[start..end]) {
                entries.push(String::from(name));
            }
        }
        start = end + 1;
    }
    Ok(entries)
}

fn read_string(path: &str) -> Result<String, Errno> {
    let fd = vfs_open(path, O_RDONLY)?;
    let mut buf = [0u8; 128];
    let n = vfs_read(fd, &mut buf)?;
    let _ = vfs_close(fd);
    let text = core::str::from_utf8(&buf[..n]).map_err(|_| Errno::EINVAL)?;
    Ok(text.trim().into())
}

fn read_hex_u16(path: &str) -> Result<u16, Errno> {
    let text = read_string(path)?;
    parse_hex(&text).map(|value| value as u16)
}

fn read_hex_u32(path: &str) -> Result<u32, Errno> {
    let text = read_string(path)?;
    parse_hex(&text).map(|value| value as u32)
}

fn parse_hex(text: &str) -> Result<u64, Errno> {
    let trimmed = text.trim();
    let digits = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    u64::from_str_radix(digits, 16).map_err(|_| Errno::EINVAL)
}
