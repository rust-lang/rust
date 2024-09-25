use std::mem;

#[allow(unused)]
#[repr(C)]
union U {
    field: (u8, u16),
}

fn main() {
    unsafe {
        let p: U = mem::transmute(0u32); // The copy when `U` is returned from `transmute` should destroy padding.
        let c = &p as *const _ as *const [u8; 4];
        // Read the entire thing, definitely contains the padding byte.
        let _val = *c;
        //~^ERROR: uninitialized
    }
}
