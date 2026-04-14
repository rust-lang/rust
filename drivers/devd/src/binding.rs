#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use alloc::string::String;

use crate::sysfs::SysDevice;

#[derive(Clone, Copy)]
pub struct Binding {
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: Option<u32>, // Added class code matching
    pub driver: &'static str,
}

const BUILTIN_BINDINGS: &[Binding] = &[
    Binding {
        vendor_id: 0x1af4,
        device_id: 0x1000,
        class_code: Some(0x020000), // Network controller
        driver: "/virtio_netd",
    },
    Binding {
        vendor_id: 0x1af4,
        device_id: 0x1050,
        class_code: Some(0x030000), // Display controller
        driver: "/virtio_gpu",
    },
    Binding {
        vendor_id: 0,
        device_id: 0,
        class_code: Some(0x010601), // AHCI Controller
        driver: "/ahci_disk",
    },
];

pub fn match_binding(device: &SysDevice) -> Option<Binding> {
    BUILTIN_BINDINGS.iter().copied().find(|binding| {
        let id_match = (binding.vendor_id == 0 || binding.vendor_id == device.vendor_id)
            && (binding.device_id == 0 || binding.device_id == device.device_id);
        let class_match = binding.class_code.map_or(true, |c| device.class_code == c);
        id_match && class_match
    })
}

pub fn mount_hint(binding: Binding, device: &SysDevice) -> Option<String> {
    if binding.driver == "/virtio_netd" && device.class_code == 0x02 {
        return Some(String::from("/dev/net/virtio0"));
    }
    None
}
