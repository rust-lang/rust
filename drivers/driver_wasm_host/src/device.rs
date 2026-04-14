#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::trace::TraceMode;
use wasmi::StoreLimits;

pub struct HostState {
    pub limits: StoreLimits,
    pub device_handle: u32,
    pub mmio: Vec<u8>, // Fake MMIO memory (4KB)
    pub trace: TraceMode,
}

impl HostState {
    pub fn new(device_handle: u32) -> Self {
        Self {
            limits: StoreLimits::default(),
            device_handle,
            mmio: vec![0u8; 4096],
            trace: TraceMode::None,
        }
    }
}
