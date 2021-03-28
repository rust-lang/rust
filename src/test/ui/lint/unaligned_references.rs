#![deny(unaligned_references)]

#[repr(packed)]
pub struct Good {
    data: u64,
    ptr: &'static u64,
    data2: [u64; 2],
    aligned: [u8; 32],
}

#[repr(packed(2))]
pub struct Packed2 {
    x: u32,
    y: u16,
    z: u8,
}

fn main() {
    unsafe {
        let good = Good { data: 0, ptr: &0, data2: [0, 0], aligned: [0; 32] };

        let _ = &good.ptr; //~ ERROR reference to packed field
        //~^ previously accepted
        let _ = &good.data; //~ ERROR reference to packed field
        //~^ previously accepted
        // Error even when turned into raw pointer immediately.
        let _ = &good.data as *const _; //~ ERROR reference to packed field
        //~^ previously accepted
        let _: *const _ = &good.data; //~ ERROR reference to packed field
        //~^ previously accepted
        // Error on method call.
        let _ = good.data.clone(); //~ ERROR reference to packed field
        //~^ previously accepted
        // Error for nested fields.
        let _ = &good.data2[0]; //~ ERROR reference to packed field
        //~^ previously accepted

        let _ = &*good.ptr; // ok, behind a pointer
        let _ = &good.aligned; // ok, has align 1
        let _ = &good.aligned[2]; // ok, has align 1
    }

    unsafe {
        let packed2 = Packed2 { x: 0, y: 0, z: 0 };
        let _ = &packed2.x; //~ ERROR reference to packed field
        //~^ previously accepted
        let _ = &packed2.y; // ok, has align 2 in packed(2) struct
        let _ = &packed2.z; // ok, has align 1
    }
}
