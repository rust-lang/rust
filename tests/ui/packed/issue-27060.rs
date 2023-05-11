#[repr(packed)]
pub struct Good {
    data: &'static u32,
    data2: [&'static u32; 2],
    aligned: [u8; 32],
}

fn main() {
    let good = Good {
        data: &0,
        data2: [&0, &0],
        aligned: [0; 32]
    };

    let _ = &good.data; //~ ERROR reference to packed field
    let _ = &good.data2[0]; //~ ERROR reference to packed field

    let _ = &good.data; //~ ERROR reference to packed field
    let _ = &good.data2[0]; //~ ERROR reference to packed field
    let _ = &*good.data; // ok, behind a pointer
    let _ = &good.aligned; // ok, has align 1
    let _ = &good.aligned[2]; // ok, has align 1
}
