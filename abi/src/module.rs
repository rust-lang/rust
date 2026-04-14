use crate::device::RootCaps;
use crate::types::ThingId;

/// The section name where the manifest is stored in the ELF binary.
pub const MANIFEST_SECTION: &str = ".thingos.manifest";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModuleKind {
    Driver = 1,
    App = 2,
    Service = 3,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ManifestMatch {
    pub compatible: [u8; 32],
}

impl ManifestMatch {
    pub const fn new(s: &str) -> Self {
        let mut bytes = [0u8; 32];
        let mut i = 0;
        let s_bytes = s.as_bytes();
        while i < s_bytes.len() && i < 32 {
            bytes[i] = s_bytes[i];
            i += 1;
        }
        Self { compatible: bytes }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ModuleManifest {
    pub magic: u64,
    pub kind: ModuleKind,
    pub name: [u8; 32],
    pub version: u32,
    pub match_count: usize,
    pub matches: [ManifestMatch; 4],
}

impl ModuleManifest {
    pub const fn new_driver(name: &str, matching: &str) -> Self {
        let mut name_bytes = [0u8; 32];
        let mut i = 0;
        let s_bytes = name.as_bytes();
        while i < s_bytes.len() && i < 32 {
            name_bytes[i] = s_bytes[i];
            i += 1;
        }

        Self {
            magic: 0xCAFEBABE,
            kind: ModuleKind::Driver,
            name: name_bytes,
            version: 1,
            match_count: 1,
            matches: [
                ManifestMatch::new(matching),
                ManifestMatch::new(""),
                ManifestMatch::new(""),
                ManifestMatch::new(""),
            ],
        }
    }
}

/// Context passed to a driver when it is spawned.
/// This pointer is passed as `arg` (the second argument to main).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DriverCtx {
    pub device_id: ThingId,
    pub root_caps: RootCaps,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_match_new_short_string() {
        let m = ManifestMatch::new("pci:v1234");
        assert_eq!(&m.compatible[..9], b"pci:v1234");
        assert_eq!(m.compatible[9], 0); // null-terminated
    }

    #[test]
    fn manifest_match_new_exact_32_bytes() {
        let s = "01234567890123456789012345678901"; // exactly 32 bytes
        let m = ManifestMatch::new(s);
        assert_eq!(&m.compatible, s.as_bytes());
    }

    #[test]
    fn manifest_match_new_truncates_long_string() {
        let s = "this string is definitely longer than thirty-two bytes";
        let m = ManifestMatch::new(s);
        assert_eq!(&m.compatible, &s.as_bytes()[..32]);
    }

    #[test]
    fn manifest_match_new_empty_string() {
        let m = ManifestMatch::new("");
        assert!(m.compatible.iter().all(|&b| b == 0));
    }

    #[test]
    fn module_manifest_new_driver_sets_kind() {
        let m = ModuleManifest::new_driver("test_drv", "pci:v8086");
        assert_eq!(m.kind, ModuleKind::Driver);
    }

    #[test]
    fn module_manifest_new_driver_sets_magic() {
        let m = ModuleManifest::new_driver("test_drv", "pci:v8086");
        assert_eq!(m.magic, 0xCAFEBABE);
    }

    #[test]
    fn module_manifest_new_driver_sets_name() {
        let m = ModuleManifest::new_driver("mydriver", "pci:v1234");
        assert_eq!(&m.name[..8], b"mydriver");
    }

    #[test]
    fn module_manifest_new_driver_sets_match() {
        let m = ModuleManifest::new_driver("drv", "pci:v8086");
        assert_eq!(m.match_count, 1);
        assert_eq!(&m.matches[0].compatible[..9], b"pci:v8086");
    }

    #[test]
    fn module_manifest_new_driver_version_is_one() {
        let m = ModuleManifest::new_driver("drv", "compat");
        assert_eq!(m.version, 1);
    }

    // Struct size/layout tests for ABI stability
    #[test]
    fn manifest_match_size() {
        assert_eq!(core::mem::size_of::<ManifestMatch>(), 32);
    }

    #[test]
    fn driver_ctx_size() {
        // ThingId(u64) + RootCaps
        assert_eq!(
            core::mem::size_of::<DriverCtx>(),
            core::mem::size_of::<ThingId>() + core::mem::size_of::<RootCaps>()
        );
    }
}
