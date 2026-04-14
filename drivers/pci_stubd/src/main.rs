#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use alloc::string::String;
use stem::{info, warn};

const MAX_TRACKED: usize = 128;

const CLASS_DISPLAY: u8 = 0x03;
const CLASS_NETWORK: u8 = 0x02;
const CLASS_SERIAL_BUS: u8 = 0x0c;
const CLASS_STORAGE: u8 = 0x01;

const SUBCLASS_NETWORK_OTHER: u8 = 0x80;
const SUBCLASS_USB: u8 = 0x03;
const SUBCLASS_NVME: u8 = 0x08;

const PROGIF_XHCI: u8 = 0x30;
const PROGIF_NVME: u8 = 0x02;

#[derive(Copy, Clone)]
struct PciRule {
    name: &'static str,
    vendor_id: u16,
    device_id: Option<u16>,
    class_code: u8,
    subclass: Option<u8>,
    prog_if: Option<u8>,
    role_kind: &'static str,
}

const RULES: [PciRule; 9] = [
    // Discrete NVIDIA mobile GPUs (exact GA107M id + class fallback for this vendor/class)
    PciRule {
        name: "nvidia-ga107m-gpu",
        vendor_id: 0x10de,
        device_id: Some(0x25a2),
        class_code: CLASS_DISPLAY,
        subclass: None,
        prog_if: None,
        role_kind: "drv.display.nvidia",
    },
    PciRule {
        name: "nvidia-display-fallback",
        vendor_id: 0x10de,
        device_id: None,
        class_code: CLASS_DISPLAY,
        subclass: None,
        prog_if: None,
        role_kind: "drv.display.nvidia",
    },
    // AMD iGPU (Rembrandt class)
    PciRule {
        name: "amd-rembrandt-igpu",
        vendor_id: 0x1002,
        device_id: Some(0x1681),
        class_code: CLASS_DISPLAY,
        subclass: None,
        prog_if: None,
        role_kind: "drv.display.amd",
    },
    // AMD Rembrandt USB4 XHCI families (known IDs + class fallback)
    PciRule {
        name: "amd-rembrandt-xhci-161d",
        vendor_id: 0x1022,
        device_id: Some(0x161d),
        class_code: CLASS_SERIAL_BUS,
        subclass: Some(SUBCLASS_USB),
        prog_if: Some(PROGIF_XHCI),
        role_kind: "drv.usb.xhci.amd",
    },
    PciRule {
        name: "amd-rembrandt-xhci-161e",
        vendor_id: 0x1022,
        device_id: Some(0x161e),
        class_code: CLASS_SERIAL_BUS,
        subclass: Some(SUBCLASS_USB),
        prog_if: Some(PROGIF_XHCI),
        role_kind: "drv.usb.xhci.amd",
    },
    PciRule {
        name: "amd-rembrandt-xhci-fallback",
        vendor_id: 0x1022,
        device_id: None,
        class_code: CLASS_SERIAL_BUS,
        subclass: Some(SUBCLASS_USB),
        prog_if: Some(PROGIF_XHCI),
        role_kind: "drv.usb.xhci.amd",
    },
    PciRule {
        name: "realtek-rtl8852be",
        vendor_id: 0x10ec,
        device_id: Some(0xb852),
        class_code: CLASS_NETWORK,
        subclass: Some(SUBCLASS_NETWORK_OTHER),
        prog_if: None,
        role_kind: "drv.net.rtl8852be",
    },
    // WD/SanDisk NVMe families from your lspci tree
    PciRule {
        name: "wd-sn740-nvme",
        vendor_id: 0x15b7,
        device_id: Some(0x5003),
        class_code: CLASS_STORAGE,
        subclass: Some(SUBCLASS_NVME),
        prog_if: Some(PROGIF_NVME),
        role_kind: "drv.storage.nvme.wd",
    },
    PciRule {
        name: "wd-nvme-fallback",
        vendor_id: 0x15b7,
        device_id: None,
        class_code: CLASS_STORAGE,
        subclass: Some(SUBCLASS_NVME),
        prog_if: Some(PROGIF_NVME),
        role_kind: "drv.storage.nvme.wd",
    },
];

fn has_device_id(device_id: u16, rule_device_id: Option<u16>) -> bool {
    match rule_device_id {
        Some(expected) => expected == device_id,
        None => true,
    }
}

fn has_subclass(subclass: u8, rule_subclass: Option<u8>) -> bool {
    match rule_subclass {
        Some(expected) => expected == subclass,
        None => true,
    }
}

fn has_prog_if(prog_if: u8, rule_prog_if: Option<u8>) -> bool {
    match rule_prog_if {
        Some(expected) => expected == prog_if,
        None => true,
    }
}

fn find_rule(
    vendor_id: u16,
    device_id: u16,
    class_code: u8,
    subclass: u8,
    prog_if: u8,
) -> Option<PciRule> {
    for rule in RULES {
        if vendor_id != rule.vendor_id {
            continue;
        }
        if class_code != rule.class_code {
            continue;
        }
        if !has_device_id(device_id, rule.device_id) {
            continue;
        }
        if !has_subclass(subclass, rule.subclass) {
            continue;
        }
        if !has_prog_if(prog_if, rule.prog_if) {
            continue;
        }
        return Some(rule);
    }
    None
}

/// Tracks claimed device sysfs paths.
type ClaimedDevices = alloc::vec::Vec<String>;

fn already_claimed(path: &str, tracked: &ClaimedDevices) -> bool {
    tracked.iter().any(|p| p == path)
}

fn push_claim(tracked: &mut ClaimedDevices, path: String) {
    if tracked.len() < MAX_TRACKED {
        tracked.push(path);
    }
}

fn scan_once(tracked: &mut ClaimedDevices) {
    use abi::syscall::vfs_flags::O_RDONLY;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_readdir};

    let fd = match vfs_open("/sys/devices", O_RDONLY) {
        Ok(fd) => fd,
        Err(_) => return,
    };
    let mut buf = [0u8; 4096];
    let n = vfs_readdir(fd, &mut buf).unwrap_or(0);
    let _ = vfs_close(fd);

    let mut offset = 0;
    while offset < n {
        let mut end = offset;
        while end < n && buf[end] != 0 {
            end += 1;
        }
        if end > offset {
            if let Ok(name) = core::str::from_utf8(&buf[offset..end]) {
                let path = alloc::format!("/sys/devices/{}", name);
                process_device(tracked, &path);
            }
        }
        offset = end + 1;
    }
}

fn process_device(tracked: &mut ClaimedDevices, path: &str) {
    if already_claimed(path, tracked) {
        return;
    }

    let vendor_id = read_sys_u32(&alloc::format!("{}/vendor", path)).unwrap_or(0) as u16;
    let device_id = read_sys_u32(&alloc::format!("{}/device", path)).unwrap_or(0) as u16;

    let class_full = read_sys_u32(&alloc::format!("{}/class", path)).unwrap_or(0);
    let class_code = (class_full >> 16) as u8;
    let subclass = (class_full >> 8) as u8;
    let prog_if = (class_full & 0xFF) as u8;

    let Some(rule) = find_rule(vendor_id, device_id, class_code, subclass, prog_if) else {
        return;
    };

    match stem::syscall::device_claim(path) {
        Ok(claim) => {
            info!(
                "pci_stubd: bound {} to {} vendor={:04x} device={:04x} class={:02x}:{:02x}:{:02x} claim={}",
                rule.name, path, vendor_id, device_id, class_code, subclass, prog_if, claim
            );
            push_claim(tracked, alloc::string::String::from(path));
        }
        Err(e) => {
            warn!("pci_stubd: failed bind {} at {}: {:?}", rule.name, path, e);
        }
    }
}

fn read_sys_u32(path: &str) -> Result<u32, abi::errors::Errno> {
    use abi::syscall::vfs_flags::O_RDONLY;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read};

    let fd = vfs_open(path, O_RDONLY)?;
    let mut buf = [0u8; 32];
    let n = vfs_read(fd, &mut buf)?;
    let _ = vfs_close(fd);

    let s = core::str::from_utf8(&buf[..n]).map_err(|_| abi::errors::Errno::EIO)?;
    let trimmed = s.trim();
    if trimmed.starts_with("0x") {
        u32::from_str_radix(&trimmed[2..], 16).map_err(|_| abi::errors::Errno::EIO)
    } else {
        trimmed.parse::<u32>().map_err(|_| abi::errors::Errno::EIO)
    }
}

#[stem::main]
fn main(_arg: usize) -> ! {
    info!("pci_stubd: starting pci-id matcher");
    let mut tracked: ClaimedDevices = alloc::vec::Vec::new();

    loop {
        scan_once(&mut tracked);
        stem::time::sleep_ms(1000);
    }
}
