use crate::{BootModuleDesc, FramebufferInfo, PhysRange};
use spin::RwLock;

pub static BOOT_INFO: RwLock<Option<BootSyscallInfo>> = RwLock::new(None);

#[derive(Clone, Copy)]
pub struct BootSyscallInfo {
    pub memory_map: &'static [PhysRange],
    pub modules: &'static [BootModuleDesc],
    pub framebuffer: Option<FramebufferInfo>,
    pub hhdm_offset: u64,
    pub acpi_rsdp: Option<u64>,
    pub dtb_ptr: Option<u64>,
}

pub fn set(info: BootSyscallInfo) {
    *BOOT_INFO.write() = Some(info);
}

pub fn get() -> Option<BootSyscallInfo> {
    *BOOT_INFO.read()
}
