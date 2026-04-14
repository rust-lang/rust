#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    unsafe {
        let mut i = 0;
        while i < n {
            *dest.add(i) = *src.add(i);
            i += 1;
        }
        dest
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    unsafe {
        if src < dest as *const u8 {
            let mut i = n;
            while i > 0 {
                i -= 1;
                *dest.add(i) = *src.add(i);
            }
        } else {
            let mut i = 0;
            while i < n {
                *dest.add(i) = *src.add(i);
                i += 1;
            }
        }
        dest
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memset(dest: *mut u8, c: i32, n: usize) -> *mut u8 {
    unsafe {
        let mut i = 0;
        while i < n {
            *dest.add(i) = c as u8;
            i += 1;
        }
        dest
    }
}

pub fn memory_map() -> &'static [kernel::PhysRange] {
    use crate::requests::MEMORY_MAP_REQUEST;
    use kernel::{PhysRange, PhysRangeKind};

    if let Some(resp) = MEMORY_MAP_REQUEST.get_response() {
        static mut RANGES: [PhysRange; 64] = [PhysRange {
            start: 0,
            end: 0,
            kind: PhysRangeKind::Other,
        }; 64];
        static mut COUNT: usize = 0;

        unsafe {
            if COUNT == 0 {
                for (i, entry) in resp.entries().into_iter().enumerate() {
                    if i >= 64 {
                        break;
                    }
                    RANGES[i] = PhysRange {
                        start: entry.base,
                        end: entry.base + entry.length,
                        kind: match entry.entry_type {
                            limine::memory_map::EntryType::USABLE => PhysRangeKind::Usable,
                            limine::memory_map::EntryType::RESERVED => PhysRangeKind::Reserved,
                            limine::memory_map::EntryType::ACPI_RECLAIMABLE => PhysRangeKind::Acpi,
                            limine::memory_map::EntryType::BOOTLOADER_RECLAIMABLE => {
                                PhysRangeKind::Reserved
                            }
                            limine::memory_map::EntryType::FRAMEBUFFER => {
                                PhysRangeKind::Framebuffer
                            }
                            _ => PhysRangeKind::Other,
                        },
                    };
                    COUNT += 1;
                }
            }
            &RANGES[..COUNT]
        }
    } else {
        &[]
    }
}
