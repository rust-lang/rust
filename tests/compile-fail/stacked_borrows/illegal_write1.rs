fn evil(x: &u32) {
    // mutating shared ref without `UnsafeCell`
    let x : *mut u32 = x as *const _ as *mut _;
    unsafe { *x = 42; }
}

fn main() {
    let target = Box::new(42); // has an implicit raw
    let ref_ = &*target;
    evil(ref_); // invalidates shared ref, activates raw
    let _x = *ref_; //~ ERROR is not frozen long enough
}
