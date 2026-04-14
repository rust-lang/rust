#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


mod binding;
mod spawn;
mod sysfs;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use binding::{match_binding, mount_hint};
use spawn::ManagedDriver;
use stem::{debug, info, warn};
use sysfs::scan_devices;

#[stem::main]
fn main(_arg: usize) -> ! {
    debug!("DEVD: starting device discovery manager");

    let mut drivers: BTreeMap<alloc::string::String, ManagedDriver> = BTreeMap::new();

    loop {
        match scan_devices() {
            Ok(devices) => reconcile_devices(&mut drivers, devices),
            Err(err) => warn!("DEVD: scan of /sys/devices failed: {:?}", err),
        }

        for managed in drivers.values_mut() {
            managed.monitor();
        }

        stem::time::sleep_ms(100);
    }
}

fn reconcile_devices(
    drivers: &mut BTreeMap<alloc::string::String, ManagedDriver>,
    devices: Vec<sysfs::SysDevice>,
) {
    let mut seen = BTreeMap::new();

    for device in devices {
        seen.insert(device.slot.clone(), ());
        if !device.present {
            continue;
        }
        let Some(binding) = match_binding(&device) else {
            continue;
        };

        let mount_path = mount_hint(binding, &device);
        let managed = drivers
            .entry(device.slot.clone())
            .or_insert_with(|| ManagedDriver::new(&device, binding, mount_path));
        managed.ensure_running();
    }

    let stale_slots: Vec<_> = drivers
        .keys()
        .filter(|slot| !seen.contains_key(*slot))
        .cloned()
        .collect();

    for slot in stale_slots {
        if let Some(mut managed) = drivers.remove(&slot) {
            warn!("DEVD: device {} disappeared", slot);
            managed.mark_removed();
        }
    }
}
