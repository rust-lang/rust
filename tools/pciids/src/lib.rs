//! PCI ID parser and code generator utilities.
//!
//! # Examples
//! ```
//! use pciids::{build_tables, lookup_device, lookup_vendor, parse_pci_ids};
//!
//! let input = "1af4 Virtio\n\t1000 Virtio Device\n";
//! let parsed = parse_pci_ids(input);
//! let tables = build_tables(&parsed.vendors);
//! assert_eq!(lookup_vendor(&tables, 0x1af4), Some("Virtio"));
//! assert_eq!(lookup_device(&tables, 0x1af4, 0x1000), Some("Virtio Device"));
//! ```

use std::fmt::Write;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SnapshotInfo {
    pub version: Option<String>,
    pub date: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub id: u16,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vendor {
    pub id: u16,
    pub name: String,
    pub devices: Vec<Device>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Full,
    Minimal,
}

impl Mode {
    pub fn from_env(val: Option<String>) -> Self {
        match val.as_deref() {
            Some("minimal") => Mode::Minimal,
            _ => Mode::Full,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Full => "full",
            Mode::Minimal => "minimal",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedPciIds {
    pub snapshot: SnapshotInfo,
    pub vendors: Vec<Vendor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VendorEntry {
    pub vendor: u16,
    pub name_off: u32,
    pub name_len: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceEntry {
    pub vendor: u16,
    pub device: u16,
    pub name_off: u32,
    pub name_len: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedTables {
    pub strings: Vec<u8>,
    pub vendors: Vec<VendorEntry>,
    pub devices: Vec<DeviceEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseIdError {
    InvalidHex,
    Overflow,
}

pub const MINIMAL_VENDOR_ALLOWLIST: &[u16] = &[
    0x8086, // Intel
    0x1af4, // Red Hat / Virtio
    0x1234, // QEMU / Bochs
    0x15ad, // VMware
    0x80ee, // VirtualBox
    0x1b36, // Red Hat
];

const MIN_INTEL_DEVICES: &[u16] = &[
    0x100e, 0x10d3, 0x1237, 0x2415, 0x2922, 0x7000, 0x7010, 0x7111, 0x7113,
];
const MIN_VIRTIO_DEVICES: &[u16] = &[
    0x1000, 0x1001, 0x1002, 0x1003, 0x1004, 0x1005, 0x1009, 0x1041, 0x1042, 0x1043, 0x1044, 0x1045,
    0x1048, 0x1049,
];
const MIN_QEMU_DEVICES: &[u16] = &[0x1111];
const MIN_VMWARE_DEVICES: &[u16] = &[0x0405, 0x0790, 0x07a0];
const MIN_VBOX_DEVICES: &[u16] = &[0xbeef];
const MIN_REDHAT_DEVICES: &[u16] = &[0x000d];

pub fn parse_pci_ids(input: &str) -> ParsedPciIds {
    let mut snapshot = SnapshotInfo::default();
    let mut vendors: Vec<Vendor> = Vec::new();

    for raw_line in input.lines() {
        let line = raw_line.trim_end();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            parse_metadata(line, &mut snapshot);
            continue;
        }
        let line = strip_inline_comment(line);
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('\t') {
            if rest.starts_with('\t') {
                // Subsystems and other nested records are ignored.
                continue;
            }
            if let Some(vendor) = vendors.last_mut() {
                if let Some((id_hex, name)) = split_id_and_name(rest.trim_start()) {
                    if let Ok(device_id) = parse_hex_id(id_hex) {
                        vendor.devices.push(Device {
                            id: device_id,
                            name: name.to_string(),
                        });
                    }
                }
            }
            continue;
        }

        if let Some((id_hex, name)) = split_id_and_name(line.trim_start()) {
            if let Ok(vendor_id) = parse_hex_id(id_hex) {
                vendors.push(Vendor {
                    id: vendor_id,
                    name: name.to_string(),
                    devices: Vec::new(),
                });
            }
        }
    }

    ParsedPciIds { snapshot, vendors }
}

pub fn filter_vendors(vendors: &[Vendor], mode: Mode) -> Vec<Vendor> {
    match mode {
        Mode::Full => vendors.to_vec(),
        Mode::Minimal => vendors
            .iter()
            .filter(|v| MINIMAL_VENDOR_ALLOWLIST.contains(&v.id))
            .cloned()
            .map(|mut v| {
                if let Some(allow) = minimal_device_allowlist(v.id) {
                    v.devices.retain(|d| allow.contains(&d.id));
                }
                v
            })
            .collect(),
    }
}

fn minimal_device_allowlist(vendor: u16) -> Option<&'static [u16]> {
    match vendor {
        0x8086 => Some(MIN_INTEL_DEVICES),
        0x1af4 => Some(MIN_VIRTIO_DEVICES),
        0x1234 => Some(MIN_QEMU_DEVICES),
        0x15ad => Some(MIN_VMWARE_DEVICES),
        0x80ee => Some(MIN_VBOX_DEVICES),
        0x1b36 => Some(MIN_REDHAT_DEVICES),
        _ => None,
    }
}

pub fn build_tables(vendors: &[Vendor]) -> GeneratedTables {
    let mut vendors_sorted: Vec<Vendor> = vendors.to_vec();

    for vendor in vendors_sorted.iter_mut() {
        vendor.devices.sort_by_key(|d| d.id);
        vendor.devices.dedup_by(|a, b| a.id == b.id);
    }

    vendors_sorted.sort_by_key(|v| v.id);
    vendors_sorted = merge_duplicate_vendors(vendors_sorted);

    let mut strings = Vec::new();
    let mut vendor_entries = Vec::new();
    let mut device_entries = Vec::new();

    for vendor in vendors_sorted.iter() {
        let (v_off, v_len) = push_name(&mut strings, &vendor.name);
        vendor_entries.push(VendorEntry {
            vendor: vendor.id,
            name_off: v_off,
            name_len: v_len,
        });

        for device in vendor.devices.iter() {
            let (d_off, d_len) = push_name(&mut strings, &device.name);
            device_entries.push(DeviceEntry {
                vendor: vendor.id,
                device: device.id,
                name_off: d_off,
                name_len: d_len,
            });
        }
    }

    GeneratedTables {
        strings,
        vendors: vendor_entries,
        devices: device_entries,
    }
}

pub fn lookup_vendor<'a>(tables: &'a GeneratedTables, vendor: u16) -> Option<&'a str> {
    tables
        .vendors
        .binary_search_by_key(&vendor, |v| v.vendor)
        .ok()
        .and_then(|idx| {
            let entry = &tables.vendors[idx];
            table_str_at(tables, entry.name_off, entry.name_len)
        })
}

pub fn lookup_device<'a>(tables: &'a GeneratedTables, vendor: u16, device: u16) -> Option<&'a str> {
    tables
        .devices
        .binary_search_by(|entry| {
            let key = (entry.vendor, entry.device);
            if key < (vendor, device) {
                core::cmp::Ordering::Less
            } else if key > (vendor, device) {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .ok()
        .and_then(|idx| {
            let entry = &tables.devices[idx];
            table_str_at(tables, entry.name_off, entry.name_len)
        })
}

pub fn render_rust(tables: &GeneratedTables, snapshot: &SnapshotInfo, mode: Mode) -> String {
    let mut out = String::new();
    writeln!(
        &mut out,
        "// @generated by tools/pciids. Do not edit by hand."
    )
    .unwrap();
    writeln!(
        &mut out,
        "pub const PCI_IDS_VERSION: &str = \"{}\";",
        snapshot.version.as_deref().unwrap_or("")
    )
    .unwrap();
    writeln!(
        &mut out,
        "pub const PCI_IDS_DATE: &str = \"{}\";",
        snapshot.date.as_deref().unwrap_or("")
    )
    .unwrap();
    writeln!(
        &mut out,
        "pub const PCI_IDS_MODE: &str = \"{}\";",
        mode.as_str()
    )
    .unwrap();
    writeln!(
        &mut out,
        "pub const PCI_IDS_NAME_SOURCE: &str = \"{}\";",
        name_source(snapshot)
    )
    .unwrap();
    writeln!(
        &mut out,
        "pub const PCI_VENDOR_COUNT: usize = {};",
        tables.vendors.len()
    )
    .unwrap();
    writeln!(
        &mut out,
        "pub const PCI_DEVICE_COUNT: usize = {};",
        tables.devices.len()
    )
    .unwrap();

    writeln!(
        &mut out,
        "#[derive(Copy, Clone)]\npub struct VendorEntry {{ pub vendor: u16, pub name_off: u32, pub name_len: u16 }}"
    )
    .unwrap();
    writeln!(
        &mut out,
        "#[derive(Copy, Clone)]\npub struct DeviceEntry {{ pub vendor: u16, pub device: u16, pub name_off: u32, pub name_len: u16 }}"
    )
    .unwrap();

    out.push_str("pub static STRINGS: &[u8] = b\"");
    escape_bytes(&tables.strings, &mut out);
    out.push_str("\";\n");

    out.push_str("pub static VENDORS: &[VendorEntry] = &[\n");
    for v in tables.vendors.iter() {
        writeln!(
            &mut out,
            "    VendorEntry {{ vendor: 0x{0:04x}, name_off: {1}, name_len: {2} }},",
            v.vendor, v.name_off, v.name_len
        )
        .unwrap();
    }
    out.push_str("];\n");

    out.push_str("pub static DEVICES: &[DeviceEntry] = &[\n");
    for d in tables.devices.iter() {
        writeln!(
            &mut out,
            "    DeviceEntry {{ vendor: 0x{0:04x}, device: 0x{1:04x}, name_off: {2}, name_len: {3} }},",
            d.vendor, d.device, d.name_off, d.name_len
        )
        .unwrap();
    }
    out.push_str("];\n");

    out.push_str(
        r#"
#[inline]
fn str_at(off: u32, len: u16) -> &'static str {
    unsafe {
        core::str::from_utf8_unchecked(&STRINGS[off as usize .. off as usize + len as usize])
    }
}

pub fn vendor_name(vendor: u16) -> Option<&'static str> {
    VENDORS
        .binary_search_by_key(&vendor, |v| v.vendor)
        .ok()
        .map(|idx| {
            let entry = &VENDORS[idx];
            str_at(entry.name_off, entry.name_len)
        })
}

pub fn device_name(vendor: u16, device: u16) -> Option<&'static str> {
    DEVICES
        .binary_search_by(|entry| {
            let key = (entry.vendor, entry.device);
            if key < (vendor, device) {
                core::cmp::Ordering::Less
            } else if key > (vendor, device) {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .ok()
        .map(|idx| {
            let entry = &DEVICES[idx];
            str_at(entry.name_off, entry.name_len)
        })
}
"#,
    );

    out
}

fn name_source(snapshot: &SnapshotInfo) -> String {
    if let Some(ver) = snapshot.version.as_ref() {
        format!("pci.ids@{}", ver)
    } else if let Some(date) = snapshot.date.as_ref() {
        let trimmed_date = date.split_whitespace().next().unwrap_or(date);
        format!("pci.ids@{}", trimmed_date)
    } else {
        "pci.ids@unknown".to_string()
    }
}

fn parse_hex_id(id_hex: &str) -> Result<u16, ParseIdError> {
    let value = u32::from_str_radix(id_hex, 16).map_err(|_| ParseIdError::InvalidHex)?;
    if value > u16::MAX as u32 {
        return Err(ParseIdError::Overflow);
    }
    Ok(value as u16)
}

fn strip_inline_comment(line: &str) -> &str {
    if let Some(idx) = line.find('#') {
        let (before, _) = line.split_at(idx);
        if before
            .chars()
            .last()
            .map(|c| c.is_whitespace())
            .unwrap_or(true)
        {
            return before.trim_end();
        }
    }
    line
}

fn merge_duplicate_vendors(vendors: Vec<Vendor>) -> Vec<Vendor> {
    let mut merged: Vec<Vendor> = Vec::new();
    for vendor in vendors.into_iter() {
        if let Some(last) = merged.last_mut() {
            if last.id == vendor.id {
                if last.name.is_empty() && !vendor.name.is_empty() {
                    last.name = vendor.name.clone();
                }
                last.devices.extend(vendor.devices.into_iter());
                last.devices.sort_by_key(|d| d.id);
                last.devices.dedup_by(|a, b| a.id == b.id);
                continue;
            }
        }
        merged.push(vendor);
    }
    merged
}

fn push_name(strings: &mut Vec<u8>, name: &str) -> (u32, u16) {
    assert!(
        strings.len() <= u32::MAX as usize,
        "PCI string table overflow"
    );
    let offset = strings.len() as u32;
    let len = name.as_bytes().len();
    assert!(len <= u16::MAX as usize, "PCI name too long to encode");
    strings.extend_from_slice(name.as_bytes());
    (offset, len as u16)
}

fn parse_metadata(line: &str, snapshot: &mut SnapshotInfo) {
    let body = line.trim_start_matches('#').trim();
    if let Some(rest) = body.strip_prefix("Version:") {
        if snapshot.version.is_none() {
            snapshot.version = Some(rest.trim().to_string());
        }
    } else if let Some(rest) = body.strip_prefix("Date:") {
        if snapshot.date.is_none() {
            snapshot.date = Some(rest.trim().to_string());
        }
    }
}

fn split_id_and_name(line: &str) -> Option<(&str, &str)> {
    let mut iter = line
        .splitn(2, char::is_whitespace)
        .filter(|s| !s.is_empty());
    let id = iter.next()?;
    let name = iter.next()?.trim_start();
    if name.is_empty() {
        None
    } else {
        Some((id, name))
    }
}

fn table_str_at<'a>(tables: &'a GeneratedTables, off: u32, len: u16) -> Option<&'a str> {
    let start = off as usize;
    let end = start + len as usize;
    let slice = tables.strings.get(start..end)?;
    std::str::from_utf8(slice).ok()
}

fn escape_bytes(bytes: &[u8], out: &mut String) {
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(b as char),
            _ => {
                write!(out, "\\x{b:02x}").unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# Version: test-version
# Date: 2024-01-01 00:00:00
1234 Sample Vendor
	1111 Sample Device
	2222 Another Device
# Comment
8086 Intel Corporation
	100e 82540EM Gigabit Ethernet Controller
	10d3 82574L Gigabit Network Connection
		0001 ignored subsystem
"#;

    const SMALL_FIXTURE: &str = r#"
# Version: v1
# Date: 2024-01-01 00:00:00
1234 Alpha
	0001 A1
	0002 A2
	0003 A3
1af4 Virtio
	1000 V1
8086 Intel
	100e I1
"#;

    #[test]
    fn parses_metadata_and_records() {
        let parsed = parse_pci_ids(SAMPLE);
        assert_eq!(parsed.snapshot.version.as_deref(), Some("test-version"));
        assert_eq!(parsed.snapshot.date.as_deref(), Some("2024-01-01 00:00:00"));
        assert_eq!(parsed.vendors.len(), 2);
        assert_eq!(parsed.vendors[0].id, 0x1234);
        assert_eq!(parsed.vendors[0].devices.len(), 2);
        assert_eq!(parsed.vendors[1].devices.len(), 2);
    }

    #[test]
    fn parses_whitespace_and_comments() {
        let input = "# Version: v2\r\n\
1234  Alpha   # vendor comment\r\n\
\t0001  A1 \t# device comment\r\n\
\t\t0000 ignored subsystem\r\n\
\r\n\
1af4\tVirtio\r\n\
\t1000\tV1\r\n";
        let parsed = parse_pci_ids(input);
        assert_eq!(parsed.snapshot.version.as_deref(), Some("v2"));
        assert_eq!(parsed.vendors.len(), 2);
        assert_eq!(parsed.vendors[0].name, "Alpha");
        assert_eq!(parsed.vendors[0].devices[0].name, "A1");
        assert_eq!(parsed.vendors[1].name, "Virtio");
    }

    #[test]
    fn filters_minimal_allowlist() {
        let parsed = parse_pci_ids(SAMPLE);
        let filtered = filter_vendors(&parsed.vendors, Mode::Minimal);
        // 0x1234 is allowed (QEMU/Bochs), 0x8086 is allowed (Intel)
        assert_eq!(filtered.len(), 2);
        let sample = filtered.iter().find(|v| v.id == 0x1234).unwrap();
        assert_eq!(sample.devices.len(), 1);
        assert_eq!(sample.devices[0].id, 0x1111);
    }

    #[test]
    fn filters_allowlist_edge_cases() {
        let vendors = vec![Vendor {
            id: 0x9999,
            name: "Unknown".to_string(),
            devices: Vec::new(),
        }];
        let filtered = filter_vendors(&vendors, Mode::Minimal);
        assert!(filtered.is_empty());
    }

    #[test]
    fn builds_tables_and_renders() {
        let parsed = parse_pci_ids(SAMPLE);
        let tables = build_tables(&parsed.vendors);

        assert_eq!(tables.vendors.len(), 2);
        assert!(tables.devices.len() >= 3);

        let vendor_name = std::str::from_utf8(
            &tables.strings[tables.vendors[0].name_off as usize
                ..tables.vendors[0].name_off as usize + tables.vendors[0].name_len as usize],
        )
        .unwrap();
        assert_eq!(vendor_name, "Sample Vendor");

        let rendered = render_rust(&tables, &parsed.snapshot, Mode::Full);
        assert!(rendered.contains("pub static STRINGS"));
        assert!(rendered.contains("pub fn device_name"));
    }

    #[test]
    fn render_is_deterministic() {
        let parsed = parse_pci_ids(SMALL_FIXTURE);
        let tables = build_tables(&parsed.vendors);
        let rendered = render_rust(&tables, &parsed.snapshot, Mode::Full);
        let expected = r#"// @generated by tools/pciids. Do not edit by hand.
pub const PCI_IDS_VERSION: &str = "v1";
pub const PCI_IDS_DATE: &str = "2024-01-01 00:00:00";
pub const PCI_IDS_MODE: &str = "full";
pub const PCI_IDS_NAME_SOURCE: &str = "pci.ids@v1";
pub const PCI_VENDOR_COUNT: usize = 3;
pub const PCI_DEVICE_COUNT: usize = 5;
#[derive(Copy, Clone)]
pub struct VendorEntry { pub vendor: u16, pub name_off: u32, pub name_len: u16 }
#[derive(Copy, Clone)]
pub struct DeviceEntry { pub vendor: u16, pub device: u16, pub name_off: u32, pub name_len: u16 }
pub static STRINGS: &[u8] = b"AlphaA1A2A3VirtioV1IntelI1";
pub static VENDORS: &[VendorEntry] = &[
    VendorEntry { vendor: 0x1234, name_off: 0, name_len: 5 },
    VendorEntry { vendor: 0x1af4, name_off: 11, name_len: 6 },
    VendorEntry { vendor: 0x8086, name_off: 19, name_len: 5 },
];
pub static DEVICES: &[DeviceEntry] = &[
    DeviceEntry { vendor: 0x1234, device: 0x0001, name_off: 5, name_len: 2 },
    DeviceEntry { vendor: 0x1234, device: 0x0002, name_off: 7, name_len: 2 },
    DeviceEntry { vendor: 0x1234, device: 0x0003, name_off: 9, name_len: 2 },
    DeviceEntry { vendor: 0x1af4, device: 0x1000, name_off: 17, name_len: 2 },
    DeviceEntry { vendor: 0x8086, device: 0x100e, name_off: 24, name_len: 2 },
];

#[inline]
fn str_at(off: u32, len: u16) -> &'static str {
    unsafe {
        core::str::from_utf8_unchecked(&STRINGS[off as usize .. off as usize + len as usize])
    }
}

pub fn vendor_name(vendor: u16) -> Option<&'static str> {
    VENDORS
        .binary_search_by_key(&vendor, |v| v.vendor)
        .ok()
        .map(|idx| {
            let entry = &VENDORS[idx];
            str_at(entry.name_off, entry.name_len)
        })
}

pub fn device_name(vendor: u16, device: u16) -> Option<&'static str> {
    DEVICES
        .binary_search_by(|entry| {
            let key = (entry.vendor, entry.device);
            if key < (vendor, device) {
                core::cmp::Ordering::Less
            } else if key > (vendor, device) {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .ok()
        .map(|idx| {
            let entry = &DEVICES[idx];
            str_at(entry.name_off, entry.name_len)
        })
}
"#;
        assert_eq!(rendered, expected);
    }

    #[test]
    fn lookup_helpers_handle_unknowns() {
        let parsed = parse_pci_ids(SMALL_FIXTURE);
        let tables = build_tables(&parsed.vendors);
        assert_eq!(lookup_vendor(&tables, 0x1234), Some("Alpha"));
        assert_eq!(lookup_device(&tables, 0x1234, 0x0001), Some("A1"));
        assert_eq!(lookup_vendor(&tables, 0x9999), None);
        assert_eq!(lookup_device(&tables, 0x1234, 0x9999), None);
    }

    #[test]
    fn build_tables_dedups_duplicates() {
        let vendors = vec![
            Vendor {
                id: 0x1234,
                name: "Alpha".to_string(),
                devices: vec![Device {
                    id: 0x0001,
                    name: "A1".to_string(),
                }],
            },
            Vendor {
                id: 0x1234,
                name: "".to_string(),
                devices: vec![Device {
                    id: 0x0001,
                    name: "A1".to_string(),
                }],
            },
        ];
        let tables = build_tables(&vendors);
        assert_eq!(tables.vendors.len(), 1);
        assert_eq!(tables.devices.len(), 1);
    }

    #[test]
    fn parse_hex_id_bounds() {
        assert_eq!(parse_hex_id("0000"), Ok(0x0000));
        assert_eq!(parse_hex_id("ffff"), Ok(0xffff));
        assert_eq!(parse_hex_id("10000"), Err(ParseIdError::Overflow));
        assert_eq!(parse_hex_id("zzzz"), Err(ParseIdError::InvalidHex));
    }
}
