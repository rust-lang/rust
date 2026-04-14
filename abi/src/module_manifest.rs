#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    Driver = 0,
    Service = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ManifestHeader {
    pub magic: u64,
    pub kind: ModuleKind, // repr(u64) makes this 8 bytes
    pub device_kind: [u8; 64],
    pub version: u32,
    pub _reserved: u32,
}

pub const MANIFEST_MAGIC: u64 = 0xBAD_D00D_CAFE_FEED;
pub const SECTION_NAME: &str = ".thing_manifest";
