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

// The normal borrows here should be preserved
fn borrow_and_cast(mut x: i32) {
    let p = &x as *const i32;
    let q = &mut x as *const i32;
    let r = &mut x as *mut i32;
}

fn main() {}

// START rustc.address_of_reborrow.SimplifyCfg-initial.after.mir
// bb0: {
//  ...
//  _5 = &raw const (*_1); // & to *const casts
//  ...
//  _7 = &raw const (*_1);
//  ...
//  _11 = &raw const (*_1);
//  ...
//  _14 = &raw const (*_1);
//  ...
//  _16 = &raw const (*_1);
//  ...
//  _17 = &raw const (*_1); // & to *const coercions
//  ...
//  _18 = &raw const (*_1);
//  ...
//  _20 = &raw const (*_1);
//  ...
//  _22 = &raw const (*_1);
// ...
//  _24 = &raw const (*_2); // &mut to *const casts
// ...
//  _26 = &raw const (*_2);
// ...
//  _30 = &raw const (*_2);
// ...
//  _33 = &raw const (*_2);
// ...
//  _34 = &raw const (*_2); // &mut to *const coercions
// ...
//  _35 = &raw const (*_2);
// ...
//  _37 = &raw const (*_2);
// ...
//  _39 = &raw const (*_2);
// ...
//  _41 = &raw mut (*_2); // &mut to *mut casts
// ...
//  _43 = &raw mut (*_2);
// ...
//  _47 = &raw mut (*_2);
// ...
//  _50 = &raw mut (*_2);
// ...
//  _51 = &raw mut (*_2); // &mut to *mut coercions
// ...
//  _52 = &raw mut (*_2);
// ...
//  _54 = &raw mut (*_2);
// ...
//  _56 = &raw mut (*_2);
// ...
// }
// END rustc.address_of_reborrow.SimplifyCfg-initial.after.mir

// START rustc.borrow_and_cast.EraseRegions.after.mir
// bb0: {
//  ...
//  _4 = &_1;
//  ...
//  _7 = &mut _1;
//  ...
//  _10 = &mut _1;
//  ...
// }
// END rustc.borrow_and_cast.EraseRegions.after.mir
