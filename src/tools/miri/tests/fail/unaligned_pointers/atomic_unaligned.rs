//@compile-flags: -Zmiri-symbolic-alignment-check -Cdebug-assertions=no
#![feature(core_intrinsics)]

fn main() {
    // Do a 4-aligned u64 atomic access. That should be UB on all platforms,
    // even if u64 only has alignment 4.
    let z = [0u32; 2];
    let zptr = &z as *const _ as *const u64;
    unsafe {
        ::std::intrinsics::atomic_load_seqcst(zptr);
        //~^ERROR: accessing memory with alignment 4, but alignment 8 is required
    }
}
