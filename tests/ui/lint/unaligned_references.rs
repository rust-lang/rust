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
        let _ = &good.data; //~ ERROR reference to packed field
        // Error even when turned into raw pointer immediately.
        let _ = &good.data as *const _; //~ ERROR reference to packed field
        let _: *const _ = &good.data; //~ ERROR reference to packed field
        // Error on method call.
        let _ = good.data.clone(); //~ ERROR reference to packed field
        // Error for nested fields.
        let _ = &good.data2[0]; //~ ERROR reference to packed field

        let _ = &*good.ptr; // ok, behind a pointer
        let _ = &good.aligned; // ok, has align 1
        let _ = &good.aligned[2]; // ok, has align 1
    }

    unsafe {
        let packed2 = Packed2 { x: 0, y: 0, z: 0 };
        let _ = &packed2.x; //~ ERROR reference to packed field
        let _ = &packed2.y; // ok, has align 2 in packed(2) struct
        let _ = &packed2.z; // ok, has align 1
    }

    unsafe {
        struct U16(u16);

        impl Drop for U16 {
            fn drop(&mut self) {
                println!("{:p}", self);
            }
        }

        struct HasDrop;

        impl Drop for HasDrop {
            fn drop(&mut self) {}
        }

        #[allow(unused)]
        struct Wrapper {
            a: U16,
            b: HasDrop,
        }
        #[allow(unused)]
        #[repr(packed(2))]
        struct Wrapper2 {
            a: U16,
            b: HasDrop,
        }

        // An outer struct with more restrictive packing than the inner struct -- make sure we
        // notice that!
        #[repr(packed)]
        struct Misalign<T>(u8, T);

        let m1 = Misalign(
            0,
            Wrapper {
                a: U16(10),
                b: HasDrop,
            },
        );
        let _ref = &m1.1.a; //~ ERROR reference to packed field

        let m2 = Misalign(
            0,
            Wrapper2 {
                a: U16(10),
                b: HasDrop,
            },
        );
        let _ref = &m2.1.a; //~ ERROR reference to packed field
    }
}
