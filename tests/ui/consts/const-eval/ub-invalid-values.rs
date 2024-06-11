const fn bool_cast(ptr: *const bool) { unsafe {
    let _val = *ptr as u32; //~ERROR: evaluation of constant value failed
    //~^ interpreting an invalid 8-bit value as a bool
}}

const _: () = {
    let v = 3_u8;
    bool_cast(&v as *const u8 as *const bool);
};

fn main() {}
