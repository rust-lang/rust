#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    RtcCmos = 1,
    Keyboard = 2,
    Mouse = 3,
    Framebuffer = 4,
    Pci = 5,
    Display = 6,
    Terminal = 7,
    /// PCM audio stream or control node under `/dev/audio/card<N>/`.
    Audio = 8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceHandle(pub u32);

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct RootCaps;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceCall {
    pub kind: DeviceKind,
    pub op: u32,
    pub in_ptr: u64,
    pub in_len: u32,
    pub out_ptr: u64,
    pub out_len: u32,
}

// RTC OPs
pub const RTC_OP_READ_TIME: u32 = 1;

// PCI DeviceCall OPs
pub const PCI_OP_ENABLE_MSI: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PciEnableMsiRequest {
    pub claim_handle: u32,
    pub requested_vectors: u16,
    pub prefer_msix: u8,
    pub _reserved: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PciEnableMsiResponse {
    pub vector: u8,
    pub irq_mode: u8, // See PCI_IRQ_MODE_* constants
    pub _reserved: [u8; 2],
}

pub const PCI_IRQ_MODE_MSI: u8 = 1;
pub const PCI_IRQ_MODE_MSIX: u8 = 2;

pub const DEVICE_IRQ_SUBSCRIBE_VECTOR: u8 = 0;
pub const DEVICE_IRQ_SUBSCRIBE_DEVICE: u8 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RtcTime {
    pub year: u16,   // e.g. 2026
    pub month: u8,   // 1-12
    pub day: u8,     // 1-31
    pub hour: u8,    // 0-23
    pub minute: u8,  // 0-59
    pub second: u8,  // 0-59
    pub weekday: u8, // 0-6
    pub flags: u8,   // Status flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abi_sizes() {
        assert_eq!(core::mem::size_of::<DeviceKind>(), 4);
        assert_eq!(core::mem::align_of::<DeviceKind>(), 4);

        assert_eq!(core::mem::size_of::<DeviceHandle>(), 4);
        assert_eq!(core::mem::align_of::<DeviceHandle>(), 4);

        assert_eq!(core::mem::size_of::<RootCaps>(), 0);
        assert_eq!(core::mem::align_of::<RootCaps>(), 1);

        assert_eq!(core::mem::size_of::<DeviceCall>(), 40);
        assert_eq!(core::mem::align_of::<DeviceCall>(), 8);

        assert_eq!(core::mem::size_of::<PciEnableMsiRequest>(), 8);
        assert_eq!(core::mem::align_of::<PciEnableMsiRequest>(), 4);

        assert_eq!(core::mem::size_of::<PciEnableMsiResponse>(), 4);
        assert_eq!(core::mem::align_of::<PciEnableMsiResponse>(), 1);

        assert_eq!(core::mem::size_of::<RtcTime>(), 10);
        assert_eq!(core::mem::align_of::<RtcTime>(), 2);
    }
}
