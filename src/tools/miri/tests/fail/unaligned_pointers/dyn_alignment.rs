// should find the bug even without, but gets masked by optimizations
//@compile-flags: -Zmiri-disable-stacked-borrows -Cdebug-assertions=no
//@normalize-stderr-test: "but found [0-9]+" -> "but found $$ALIGN"

#[repr(align(256))]
#[derive(Debug)]
struct MuchAlign;

fn main() {
    // Try many times as this might work by chance.
    for _ in 0..20 {
        let buf = [0u32; 256];
        // `buf` is sufficiently aligned for `layout.align` on a `dyn Debug`, but not
        // for the actual alignment required by `MuchAlign`.
        // We craft a wide reference `&dyn Debug` with the vtable for `MuchAlign`. That should be UB,
        // as the reference is not aligned to its dynamic alignment requirements.
        let mut ptr = &MuchAlign as &dyn std::fmt::Debug;
        // Overwrite the data part of `ptr` so it points to `buf`.
        unsafe {
            (&mut ptr as *mut _ as *mut *const u8).write(&buf as *const _ as *const u8);
        }
        // Re-borrow that. This should be UB.
        let _ptr = &*ptr; //~ERROR: required 256 byte alignment
    }
}
