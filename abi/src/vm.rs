//! Virtual memory syscall types.
//! Must be #[repr(C)] to ensure stable layout.

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VmProt(pub u32);

impl VmProt {
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const EXEC: Self = Self(1 << 2);
    pub const USER: Self = Self(1 << 3);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl core::ops::BitOr for VmProt {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for VmProt {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VmMapFlags(pub u32);

impl VmMapFlags {
    pub const FIXED: Self = Self(1 << 0);
    pub const GUARD: Self = Self(1 << 1);
    pub const PRIVATE: Self = Self(1 << 2);
    pub const SHARED: Self = Self(1 << 3);
    pub const NORESERVE: Self = Self(1 << 4);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl core::ops::BitOr for VmMapFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for VmMapFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum VmBacking {
    Anonymous { zeroed: bool },
    File { fd: u32, offset: u64 },
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VmMapReq {
    pub addr_hint: usize,
    pub len: usize,
    pub prot: VmProt,
    pub flags: VmMapFlags,
    pub backing: VmBacking,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VmMapResp {
    pub addr: usize,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VmUnmapReq {
    pub addr: usize,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VmUnmapResp {
    pub unmapped_len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VmProtectReq {
    pub addr: usize,
    pub len: usize,
    pub prot: VmProt,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VmProtectResp;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VmAdviseReq {
    pub addr: usize,
    pub len: usize,
    pub advise: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VmAdviseResp;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VmQueryReq {
    pub addr: usize,
    pub out_ptr: usize,
    pub out_len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VmQueryResp {
    pub bytes_written: usize,
    pub more: bool,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmBackingKind {
    Anonymous = 0,
    Guard = 1,
    File = 2,
    Unknown = 255,
}

impl Default for VmBackingKind {
    fn default() -> Self {
        VmBackingKind::Unknown
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VmRegionInfo {
    pub start: usize,
    pub end: usize,
    pub prot: VmProt,
    pub flags: VmMapFlags,
    pub backing_kind: VmBackingKind,
    pub _reserved: [u8; 7],
}
