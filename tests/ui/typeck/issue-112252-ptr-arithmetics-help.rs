//@ run-rustfix

fn main() {
    let _ptr1: *const u32 = std::ptr::null();
    let _ptr2: *const u32 = std::ptr::null();
    let _a = _ptr1 + 5; //~ ERROR cannot add
    let _b = _ptr1 - 5; //~ ERROR cannot subtract
    let _c = _ptr2 - _ptr1; //~ ERROR cannot subtract
    let _d = _ptr1[5]; //~ ERROR cannot index
}
