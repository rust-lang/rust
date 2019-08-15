// Even with intptrcast and without validation, we want to be *sure* to catch bugs
// that arise from pointers being insufficiently aligned. The only way to achieve
// that is not not let programs exploit integer information for alignment, so here
// we test that this is indeed the case.
fn main() {
    let x = &mut [0u8; 3];
    let base_addr = x as *mut _ as usize;
    let base_addr_aligned = if base_addr % 2 == 0 { base_addr } else { base_addr+1 };
    let u16_ptr = base_addr_aligned as *mut u16;
    unsafe { *u16_ptr = 2; } //~ ERROR tried to access memory with alignment 1, but alignment 2 is required
    println!("{:?}", x);
}
