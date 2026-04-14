#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::string::String;

pub struct DeviceLedger {
    units: BTreeMap<String, u32>,
}

impl DeviceLedger {
    pub fn new() -> Self {
        Self {
            units: BTreeMap::new(),
        }
    }

    pub fn get(&self, class: &str) -> Option<&u32> {
        self.units.get(class)
    }

    pub fn insert(&mut self, class: String, unit: u32) {
        self.units.insert(class, unit);
    }
}
