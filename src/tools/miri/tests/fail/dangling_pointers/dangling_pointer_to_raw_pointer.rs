use std::ptr;

fn direct_raw(x: *const (i32, i32)) -> *const i32 {
    unsafe { &raw const (*x).0 }
}

// Ensure that if a raw pointer is created via an intermediate
// reference, we catch that. (Just in case someone decides to
// desugar this differently or so.)
fn via_ref(x: *const (i32, i32)) -> *const i32 {
    unsafe { &(*x).0 as *const i32 } //~ERROR: dangling pointer
}

fn main() {
    let ptr = ptr::without_provenance(0x10);
    direct_raw(ptr); // this is fine
    via_ref(ptr); // this is not
}
