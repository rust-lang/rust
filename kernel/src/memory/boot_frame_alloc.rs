use crate::PhysRange;
use crate::PhysRangeKind;

pub fn init(map: &'static [PhysRange], hhdm: u64) -> &'static mut [u64] {
    let mut largest_start = 0;
    let mut largest_len = 0;
    let mut total_memory = 0;

    for range in map {
        if range.kind == PhysRangeKind::Usable {
            let len = range.end - range.start;
            if len > largest_len {
                largest_len = len;
                largest_start = range.start;
            }
            if range.end > total_memory {
                total_memory = range.end;
            }
        }
    }

    let pages = (total_memory + 4095) / 4096;
    let bitmap_u64s = (pages as usize + 63) / 64;
    let bitmap_size = bitmap_u64s * 8;

    if largest_len < bitmap_size as u64 {
        panic!(
            "No room for frame allocator bitmap (need {} bytes, have {} bytes)",
            bitmap_size, largest_len
        );
    }

    let bitmap_ptr = (largest_start + hhdm) as *mut u64;
    unsafe {
        core::ptr::write_bytes(bitmap_ptr, 0, bitmap_size);
        core::slice::from_raw_parts_mut(bitmap_ptr, bitmap_u64s)
    }
}
