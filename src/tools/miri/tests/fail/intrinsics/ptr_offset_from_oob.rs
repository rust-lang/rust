fn main() {
    let mem = [0u8; 1];
    let ptr = mem.as_ptr();
    unsafe {
        ptr.wrapping_add(4).offset_from(ptr); //~ERROR: the memory range between them is not in-bounds of an allocation
    }
}
