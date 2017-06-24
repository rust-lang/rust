fn main() {
    let x = 2usize as *const u32;
    // This must fail because alignment is violated
    let _ = unsafe { &*x }; //~ ERROR: tried to access memory with alignment 2, but alignment 4 is required
}
