const fn bool_cast(ptr: *const bool) { unsafe {
    let _val = *ptr as u32; //~ NOTE inside `bool_cast`
    //~^ NOTE the failure occurred here
}}

const _: () = {
    let v = 3_u8;
    bool_cast(&v as *const u8 as *const bool); //~ NOTE: failed inside this call
    //~^ ERROR interpreting an invalid 8-bit value as a bool
};

fn main() {}
