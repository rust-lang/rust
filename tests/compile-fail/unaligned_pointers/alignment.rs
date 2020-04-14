fn main() {
    let mut x = [0u8; 20];
    let x_ptr: *mut u8 = x.as_mut_ptr();
    // At least one of these is definitely unaligned.
    // Currently, we guarantee to complain about the first one already (https://github.com/rust-lang/miri/issues/1074).
    unsafe {
        *(x_ptr as *mut u64) = 42; //~ ERROR accessing memory with alignment 1, but alignment
        *(x_ptr.add(1) as *mut u64) = 42;
    }
    panic!("unreachable in miri");
}
