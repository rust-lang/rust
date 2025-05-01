//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@error-in-other-file: pointer not dereferenceable

fn main() {
    unsafe {
        let ptr = Box::into_raw(Box::new(0u16));
        drop(Box::from_raw(ptr as *mut u32));
    }
}
