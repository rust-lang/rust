// Make sure that dropping types with no drop glue is DB even for invalid pointers.

fn main() {
    unsafe {
        core::ptr::drop_in_place::<u8>(core::ptr::null_mut());
    }
}
