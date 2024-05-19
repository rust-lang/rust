//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
// This tests that the size of Option<Box<i32>> is the same as *const i32.
fn option_box_deref() -> i32 {
    let val = Some(Box::new(42));
    unsafe {
        let ptr: *const i32 = std::mem::transmute::<Option<Box<i32>>, *const i32>(val);
        let ret = *ptr;
        // unleak memory
        std::mem::transmute::<*const i32, Option<Box<i32>>>(ptr);
        ret
    }
}

fn main() {
    assert_eq!(option_box_deref(), 42);
}
