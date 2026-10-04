//@compile-flags: -Zmiri-symbolic-alignment-check -Zmiri-permissive-provenance -Cdebug-assertions=no
// With the symbolic alignment check, even with intptrcast and without
// validation, we want to be *sure* to catch bugs that arise from pointers being
// insufficiently aligned. The only way to achieve that is not to let programs
// exploit integer information for alignment, so here we test that this is
// indeed the case.
//
// See https://github.com/rust-lang/miri/issues/1074.
fn main() {
    let x = &mut [0u8; 3];
    let base_addr = x as *mut _ as usize;
    // Manually make sure the pointer is properly aligned.
    let base_addr_aligned = if base_addr % 2 == 0 { base_addr } else { base_addr + 1 };
    let u16_ptr = base_addr_aligned as *mut u16;
    unsafe { *u16_ptr = 2 }; //~ERROR: with alignment 1, but alignment 2 is required
    println!("{:?}", x);
}
