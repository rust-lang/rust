fn evil(x: &u32) {
    let x : &mut u32 = unsafe { &mut *(x as *const _ as *mut _) };
    *x = 42; // mutating shared ref without `UnsafeCell`
}

fn main() {
    let target = 42;
    let ref_ = &target;
    evil(ref_); // invalidates shared ref
    let _x = *ref_; //~ ERROR should be frozen
}
