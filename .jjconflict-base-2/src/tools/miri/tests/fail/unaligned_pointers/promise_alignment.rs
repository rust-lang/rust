//@compile-flags: -Zmiri-symbolic-alignment-check
//@revisions: call_unaligned_ptr read_unaligned_ptr

#[path = "../../utils/mod.rs"]
mod utils;

#[repr(align(8))]
#[derive(Copy, Clone)]
struct Align8(#[allow(dead_code)] u64);

fn main() {
    let buffer = [0u32; 128]; // get some 4-aligned memory
    let buffer = buffer.as_ptr();
    // "Promising" the alignment down to 1 must not hurt.
    unsafe { utils::miri_promise_symbolic_alignment(buffer.cast(), 1) };
    let _val = unsafe { buffer.read() };

    // Let's find a place to promise alignment 8.
    let align8 = if buffer.addr() % 8 == 0 { buffer } else { buffer.wrapping_add(1) };
    assert!(align8.addr() % 8 == 0);
    unsafe { utils::miri_promise_symbolic_alignment(align8.cast(), 8) };
    // Promising the alignment down to 1 *again* still must not hurt.
    unsafe { utils::miri_promise_symbolic_alignment(buffer.cast(), 1) };
    // Now we can do 8-aligned reads here.
    let _val = unsafe { align8.cast::<Align8>().read() };

    // Make sure we error if the pointer is not actually aligned.
    if cfg!(call_unaligned_ptr) {
        unsafe { utils::miri_promise_symbolic_alignment(align8.add(1).cast(), 8) };
        //~[call_unaligned_ptr]^ ERROR: pointer is not actually aligned
    }

    // Also don't accept even higher-aligned reads.
    if cfg!(read_unaligned_ptr) {
        #[repr(align(16))]
        #[derive(Copy, Clone)]
        struct Align16(#[allow(dead_code)] u128);

        let align16 = if align8.addr() % 16 == 0 { align8 } else { align8.wrapping_add(2) };
        assert!(align16.addr() % 16 == 0);

        let _val = unsafe { align8.cast::<Align16>().read() };
        //~[read_unaligned_ptr]^ ERROR: accessing memory based on pointer with alignment 8, but alignment 16 is required
    }
}
