// Begin of ARM32 shim
// The raw content of this file should be processed by `generate-windows-sys`
// to be merged with the generated binding. It is not supposed to be used as
// a normal Rust module.
cfg_if::cfg_if! {
if #[cfg(target_arch = "arm")] {
#[repr(C)]
pub struct WSADATA {
    pub wVersion: u16,
    pub wHighVersion: u16,
    pub szDescription: [u8; 257],
    pub szSystemStatus: [u8; 129],
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: PSTR,
}
pub enum CONTEXT {}
}
}
// End of ARM32 shim
