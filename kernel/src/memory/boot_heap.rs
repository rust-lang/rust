use crate::{BootRuntime, BootTasking, MapKind, MapPerms, runtime};

pub struct BootHeap {
    pub base: u64,
    pub size: usize,
    pub used: usize,
}

static mut BOOT_HEAP: BootHeap = BootHeap {
    base: 0xffffffffd0000000,
    size: 64 * 1024 * 1024,
    used: 0,
};

pub fn alloc_page<R: BootRuntime>() -> u64 {
    unsafe {
        if BOOT_HEAP.used + 4096 > BOOT_HEAP.size {
            panic!("Boot heap overflow");
        }
        let virt = BOOT_HEAP.base + BOOT_HEAP.used as u64;
        BOOT_HEAP.used += 4096;

        let rt = runtime::<R>();
        let tasking = rt.tasking();
        let frame = super::alloc_frame().expect("No frames for boot heap");

        tasking
            .map_page(
                tasking.active_address_space(),
                virt,
                frame,
                MapPerms {
                    read: true,
                    write: true,
                    user: false,
                    exec: false,
                    kind: MapKind::Normal,
                },
                MapKind::Normal,
                &DumbAlloc,
            )
            .expect("Failed to map boot heap page");

        virt
    }
}

struct DumbAlloc;
impl crate::FrameAllocatorHook for DumbAlloc {
    fn alloc_frame(&self) -> Option<u64> {
        None
    }
}
