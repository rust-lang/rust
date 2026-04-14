use super::SCHEDULER;
use super::types::Scheduler;
use crate::memory::mappings::MappingList;
use crate::{BootRuntime, BootTasking, MapKind, MapPerms};
use abi::errors::Errno;
use abi::vm::{VmProt, VmRegionInfo};
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use core::sync::atomic::{AtomicPtr, Ordering};

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY_MAPPING: AtomicPtr<Mutex<MappingList>> = AtomicPtr::new(core::ptr::null_mut());
pub static CURRENT_MAPPINGS: [AtomicPtr<Mutex<MappingList>>; 32] = [EMPTY_MAPPING; 32];

pub fn add_user_mapping<R: BootRuntime>(region: VmRegionInfo) -> Result<(), Errno> {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let res = (|| {
        let ptr = match *lock {
            Some(ptr) => ptr,
            None => return Err(Errno::ENOSYS),
        };
        let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
        let cpu = super::current_cpu_index::<R>();
        let current_id = match sched.state.per_cpu.get(cpu).and_then(|pc| pc.current) {
            Some(id) => id,
            None => return Err(Errno::ESRCH),
        };

        if let Some(task) = crate::task::registry::get_task::<R>(current_id) {
            let mut mappings = task.mappings.lock();
            mappings.insert(region);
            Ok(())
        } else {
            Err(Errno::ESRCH)
        }
    })();
    rt.irq_restore(_irq);
    res
}

pub fn remove_user_mappings<R: BootRuntime>(
    addr: usize,
    len: usize,
) -> Result<Vec<(usize, usize)>, Errno> {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let res = (|| {
        let ptr = match *lock {
            Some(ptr) => ptr,
            None => return Err(Errno::ENOSYS),
        };
        let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
        let cpu = super::current_cpu_index::<R>();
        let current_id = match sched.state.per_cpu.get(cpu).and_then(|pc| pc.current) {
            Some(id) => id,
            None => return Err(Errno::ESRCH),
        };

        if let Some(task) = crate::task::registry::get_task::<R>(current_id) {
            let mut mappings = task.mappings.lock();
            Ok(mappings.remove(addr, len))
        } else {
            Err(Errno::ESRCH)
        }
    })();
    rt.irq_restore(_irq);
    res
}

pub fn check_user_mapping<R: BootRuntime>(addr: usize, len: usize, write: bool) -> bool {
    let cpu = super::current_cpu_index::<R>();
    let ptr = CURRENT_MAPPINGS[cpu].load(Ordering::Acquire);
    if ptr.is_null() {
        false
    } else {
        unsafe { (*ptr).lock().check(addr, len, write) }
    }
}

pub fn get_user_mapping_at<R: BootRuntime>(addr: usize) -> Option<VmRegionInfo> {
    let cpu = super::current_cpu_index::<R>();
    let ptr = CURRENT_MAPPINGS[cpu].load(Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        unsafe { (*ptr).lock().find_at(addr) }
    }
}

pub unsafe fn translate_user_page<R: BootRuntime>(addr: u64) -> Option<u64> {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let res = if let Some(ptr) = *lock {
        let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
        let cpu = super::current_cpu_index::<R>();
        let current_id = match sched.state.per_cpu.get(cpu).and_then(|pc| pc.current) {
            Some(id) => id,
            None => {
                rt.irq_restore(_irq);
                return None;
            }
        };

        if let Some(task) = crate::task::registry::get_task::<R>(current_id) {
            rt.tasking().translate(task.aspace, addr)
        } else {
            None
        }
    } else {
        None
    };
    rt.irq_restore(_irq);
    res
}

pub fn protect_user_range<R: BootRuntime>(
    addr: u64,
    len: usize,
    prot: VmProt,
) -> Result<(), Errno> {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let res = (|| {
        let ptr = match *lock {
            Some(ptr) => ptr,
            None => return Err(Errno::ENOSYS),
        };
        let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
        let cpu = super::current_cpu_index::<R>();
        let current_id = match sched.state.per_cpu.get(cpu).and_then(|pc| pc.current) {
            Some(id) => id,
            None => return Err(Errno::ESRCH),
        };

        if let Some(task) = crate::task::registry::get_task::<R>(current_id) {
            let mut mappings = task.mappings.lock();

            // 1. Validate range
            if !mappings.check(addr as usize, len, false) {
                return Err(Errno::EINVAL);
            }

            // 2. Perform metadata update
            mappings.protect(addr as usize, len, prot);

            // 3. Hardware update
            let perms = MapPerms {
                user: true,
                read: prot.contains(VmProt::READ),
                write: prot.contains(VmProt::WRITE),
                exec: prot.contains(VmProt::EXEC),
                kind: MapKind::Normal,
            };

            for page_addr in (addr..addr + len as u64).step_by(4096) {
                unsafe {
                    crate::memory::protect_user_page(page_addr, perms)?;
                }
            }

            Ok(())
        } else {
            Err(Errno::ESRCH)
        }
    })();
    rt.irq_restore(_irq);
    res
}
