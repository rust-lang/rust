use std::mem;

#[repr(C)]
struct Pair(u8, u16);

fn main() {
    unsafe {
        let p: Pair = mem::transmute(0u32); // The copy when `Pair` is returned from `transmute` should destroy padding.
        let c = &p as *const _ as *const u8;
        // Read the padding byte.
        let _val = *c.add(1);
        //~^ERROR: uninitialized
    }
}
