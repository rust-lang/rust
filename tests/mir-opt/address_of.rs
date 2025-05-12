// skip-filecheck
// EMIT_MIR address_of.address_of_reborrow.SimplifyCfg-initial.after.mir

fn address_of_reborrow() {
    let y = &[0; 10];
    let mut z = &mut [0; 10];

    y as *const _;
    y as *const [i32; 10];
    y as *const dyn Send;
    y as *const [i32];
    y as *const i32; // This is a cast, not a coercion

    let p: *const _ = y;
    let p: *const [i32; 10] = y;
    let p: *const dyn Send = y;
    let p: *const [i32] = y;

    z as *const _;
    z as *const [i32; 10];
    z as *const dyn Send;
    z as *const [i32];

    let p: *const _ = z;
    let p: *const [i32; 10] = z;
    let p: *const dyn Send = z;
    let p: *const [i32] = z;

    z as *mut _;
    z as *mut [i32; 10];
    z as *mut dyn Send;
    z as *mut [i32];

    let p: *mut _ = z;
    let p: *mut [i32; 10] = z;
    let p: *mut dyn Send = z;
    let p: *mut [i32] = z;
}

// The normal borrows here should be preserved
// EMIT_MIR address_of.borrow_and_cast.SimplifyCfg-initial.after.mir
fn borrow_and_cast(mut x: i32) {
    let p = &x as *const i32;
    let q = &mut x as *const i32;
    let r = &mut x as *mut i32;
}

fn main() {}
