
fn address_of_borrow() {
    let mut arr = [0; 10];

    &arr as *const _;
    &arr as *const [i32; 10];
    &arr as *const dyn Send;
    &arr as *const [i32];
    &arr as *const i32;         // This is a cast, not a coercion

    let p: *const _ = &arr;
    let p: *const [i32; 10] = &arr;
    let p: *const dyn Send = &arr;
    let p: *const [i32] = &arr;

    &mut arr as *const _;
    &mut arr as *const [i32; 10];
    &mut arr as *const dyn Send;
    &mut arr as *const [i32];

    let p: *const _ = &mut arr;
    let p: *const [i32; 10] = &mut arr;
    let p: *const dyn Send = &mut arr;
    let p: *const [i32] = &mut arr;

    &mut arr as *mut _;
    &mut arr as *mut [i32; 10];
    &mut arr as *mut dyn Send;
    &mut arr as *mut [i32];

    let p: *mut _ = &mut arr;
    let p: *mut [i32; 10] = &mut arr;
    let p: *mut dyn Send = &mut arr;
    let p: *mut [i32] = &mut arr;
}

fn address_of_reborrow() {
    let y = &[0; 10];
    let mut z = &mut [0; 10];

    y as *const _;
    y as *const [i32; 10];
    y as *const dyn Send;
    y as *const [i32];
    y as *const i32;            // This is a cast, not a coercion

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

fn main() {}

// START rustc.address_of_borrow.SimplifyCfg-initial.after.mir
// bb0: {
// ...
//  _3 = &const raw _1; // & to *const casts
// ...
//  _5 = &const raw _1;
// ...
//  _9 = &const raw _1;
// ...
//  _12 = &const raw _1;
// ...
//  _14 = &const raw _1;
// ...
//  _15 = &const raw _1; // & to *const coercions
// ...
//  _16 = &const raw _1;
// ...
//  _18 = &const raw _1;
// ...
//  _20 = &const raw _1;
// ...
//  _23 = &mut raw _1; // &mut to *const casts
// ...
//  _26 = &mut raw _1;
// ...
//  _31 = &mut raw _1;
// ...
//  _35 = &mut raw _1;
// ...
//  _37 = &mut raw _1; // &mut to *const coercions
// ...
//  _39 = &mut raw _1;
// ...
//  _42 = &mut raw _1;
// ...
//  _45 = &mut raw _1;
// ...
//  _47 = &mut raw _1; // &mut to *mut casts
// ...
//  _49 = &mut raw _1;
// ...
//  _53 = &mut raw _1;
// ...
//  _56 = &mut raw _1;
// ...
//  _57 = &mut raw _1; // &mut to *mut coercions
// ...
//  _58 = &mut raw _1;
// ...
//  _60 = &mut raw _1;
// ...
//  _62 = &mut raw _1;
// }
// END rustc.address_of_borrow.SimplifyCfg-initial.after.mir

// START rustc.address_of_reborrow.SimplifyCfg-initial.after.mir
// bb0: {
//  ...
//  _5 = &const raw (*_1); // & to *const casts
//  ...
//  _7 = &const raw (*_1);
//  ...
//  _11 = &const raw (*_1);
//  ...
//  _14 = &const raw (*_1);
//  ...
//  _16 = &const raw (*_1);
//  ...
//  _17 = &const raw (*_1); // & to *const coercions
//  ...
//  _18 = &const raw (*_1);
//  ...
//  _20 = &const raw (*_1);
//  ...
//  _22 = &const raw (*_1);
// ...
//  _24 = &const raw (*_2); // &mut to *const casts
// ...
//  _26 = &const raw (*_2);
// ...
//  _30 = &const raw (*_2);
// ...
//  _33 = &const raw (*_2);
// ...
//  _34 = &const raw (*_2); // &mut to *const coercions
// ...
//  _35 = &const raw (*_2);
// ...
//  _37 = &const raw (*_2);
// ...
//  _39 = &const raw (*_2);
// ...
//  _41 = &mut raw (*_2); // &mut to *mut casts
// ...
//  _43 = &mut raw (*_2);
// ...
//  _47 = &mut raw (*_2);
// ...
//  _50 = &mut raw (*_2);
// ...
//  _51 = &mut raw (*_2); // &mut to *mut coercions
// ...
//  _52 = &mut raw (*_2);
// ...
//  _54 = &mut raw (*_2);
// ...
//  _56 = &mut raw (*_2);
// ...
// }
// END rustc.address_of_reborrow.SimplifyCfg-initial.after.mir
