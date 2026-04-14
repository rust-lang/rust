#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use abi::ids::HandleId;
use abi::module_manifest::{ManifestHeader, ModuleKind, MANIFEST_MAGIC, SECTION_NAME};
use abi::schema::kinds;
use alloc::collections::BTreeMap;
use alloc::string::{String};
use stem::info;
use stem::syscall::vfs::{vfs_close, vfs_open};
use stem::thing::sys as thingsys;
use stem::thing::ThingId;

pub struct Registry {
    drivers: BTreeMap<String, String>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            drivers: BTreeMap::new(),
        }
    }

    pub fn scan(&mut self) {
        info!("SPROUT: Scanning boot modules via /bin...");
        let fd = match vfs_open("/bin", abi::syscall::vfs_flags::O_RDONLY) {
            Ok(fd) => fd,
            Err(_) => {
                info!("SPROUT: Failed to open /bin");
                return;
            }
        };

        let mut buf = [0u8; 4096];
        let n = match stem::syscall::vfs::vfs_readdir(fd, &mut buf) {
            Ok(n) => n,
            Err(_) => {
                let _ = vfs_close(fd);
                return;
            }
        };
        let _ = vfs_close(fd);

        let mut offset = 0usize;
        while offset < n {
            let mut end = offset;
            while end < n && buf[end] != 0 {
                end += 1;
            }
            if end > offset {
                if let Ok(name) = core::str::from_utf8(&buf[offset..end]) {
                    if name != "." && name != ".." {
                        self.scan_module_name(name);
                    }
                }
            }
            offset = end.saturating_add(1);
        }

        info!(
            "SPROUT: Registry scan complete. Found {} drivers.",
            self.drivers.len()
        );
    }

    fn scan_module_name(&mut self, mod_name: &str) {
        let path = alloc::format!("/bin/{}", mod_name);
        if let Ok(fd) = vfs_open(&path, abi::syscall::vfs_flags::O_RDONLY) {
            if let Some(header) = self.read_manifest(fd) {
                if let ModuleKind::Driver = header.kind {
                    let raw = &header.device_kind;
                    let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
                    if let Ok(dk_str) = core::str::from_utf8(&raw[..end]) {
                        info!("SPROUT: Registering driver '{}' -> '{}'", dk_str, mod_name);
                        self.drivers
                            .insert(dk_str.to_string(), mod_name.to_string());
                    }
                }
            }
            let _ = stem::syscall::vfs::vfs_close(fd);
        } else {
            // Fallback for v0 if parsing fails
            if mod_name.contains("rtc_cmos") {
                info!(
                    "SPROUT: Registering driver 'dev.rtc.Cmos' -> '{}' (fallback)",
                    mod_name
                );
                self.drivers
                    .insert("dev.rtc.Cmos".to_string(), mod_name.to_string());
            }
        }
    }

    fn read_manifest(&self, fd: u32) -> Option<ManifestHeader> {
        let mut hdr_buf = [0u8; 64];
        // Note: thingsys::read currently doesn't support offset, so we either need seek or just read sequentially.
        // For ELF header (first 64 bytes), sequential is fine.
        if stem::syscall::vfs::vfs_read(fd, &mut hdr_buf).is_err() {
            return None;
        }

        if &hdr_buf[0..4] != b"\x7fELF" {
            return None;
        }

        let shoff = u64::from_le_bytes(hdr_buf[0x28..0x30].try_into().unwrap()) as usize;
        let shentsize = u16::from_le_bytes(hdr_buf[0x3A..0x3C].try_into().unwrap()) as usize;
        let shnum = u16::from_le_bytes(hdr_buf[0x3C..0x3E].try_into().unwrap()) as usize;
        let shstrndx = u16::from_le_bytes(hdr_buf[0x3E..0x40].try_into().unwrap()) as usize;

        let strtab_sh_off = shoff + (shstrndx as usize * shentsize);
        let (strtab_off, _) = self.read_sh_info(fd, strtab_sh_off)?;

        for i in 0..shnum {
            let off = shoff + (i * shentsize);
            if let Some((sh_name_idx, sh_offset, sh_size)) = self.read_sh_entry(fd, off) {
                if let Some(name) = self.read_string(fd, strtab_off, sh_name_idx as usize) {
                    if name == SECTION_NAME {
                        let mut m_buf = [0u8; core::mem::size_of::<ManifestHeader>()];
                        if m_buf.len() > sh_size {
                            return None;
                        }
                        if self.vfs_read_at(fd, sh_offset, &mut m_buf).is_ok() {
                            let m: ManifestHeader = unsafe { core::mem::transmute(m_buf) };
                            if m.magic == MANIFEST_MAGIC {
                                return Some(m);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn vfs_read_at(
        &self,
        fd: u32,
        offset: usize,
        buf: &mut [u8],
    ) -> Result<usize, abi::errors::Errno> {
        let _ = stem::syscall::vfs::vfs_seek(fd, offset as i64, 0)?; // 0 = SEEK_SET
        stem::syscall::vfs::vfs_read(fd, buf)
    }

    fn read_sh_info(&self, fd: u32, offset: usize) -> Option<(usize, usize)> {
        let mut buf = [0u8; 64];
        if self.vfs_read_at(fd, offset, &mut buf).is_err() {
            return None;
        }
        let sh_offset = u64::from_le_bytes(buf[0x18..0x20].try_into().ok()?) as usize;
        let sh_size = u64::from_le_bytes(buf[0x20..0x28].try_into().ok()?) as usize;
        Some((sh_offset, sh_size))
    }

    fn read_sh_entry(&self, fd: u32, offset: usize) -> Option<(u32, usize, usize)> {
        let mut buf = [0u8; 64];
        if self.vfs_read_at(fd, offset, &mut buf).is_err() {
            return None;
        }
        let sh_name = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        let sh_offset = u64::from_le_bytes(buf[0x18..0x20].try_into().ok()?) as usize;
        let sh_size = u64::from_le_bytes(buf[0x20..0x28].try_into().ok()?) as usize;
        Some((sh_name, sh_offset, sh_size))
    }

    fn read_string(&self, fd: u32, strtab_off: usize, idx: usize) -> Option<String> {
        let mut buf = [0u8; 32];
        let _ = self.vfs_read_at(fd, strtab_off + idx, &mut buf);
        let end = buf.iter().position(|&c| c == 0).unwrap_or(0);
        if end == 0 {
            return None;
        }
        core::str::from_utf8(&buf[..end])
            .ok()
            .map(|s| s.to_string())
    }

    pub fn find_driver(&self, device_kind: &str) -> Option<&str> {
        self.drivers.get(device_kind).map(|s| s.as_str())
    }
}
