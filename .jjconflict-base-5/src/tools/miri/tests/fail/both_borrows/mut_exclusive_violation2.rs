//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::ptr::NonNull;

fn main() {
    unsafe {
        let x = &mut 0;
        let mut ptr1 = NonNull::from(x);
        let mut ptr2 = ptr1.clone();
        let raw1 = ptr1.as_mut();
        let raw2 = ptr2.as_mut();
        let _val = *raw1; //~[stack] ERROR: /read access .* tag does not exist in the borrow stack/
        *raw2 = 2;
        *raw1 = 3; //~[tree] ERROR: /write access through .* is forbidden/
    }
}
