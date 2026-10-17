//@ run-pass
#![feature(clone_from_ref)]

use std::alloc::System;
use std::sync::Arc;
use std::rc::Rc;

fn main() {
    let text = "Outreachy Allocators 1.0";

    // 1. & 2. Happy Path & Type Flexibility (Box with str and slice)
    let boxed_str = Box::try_clone_from_ref_in(text, System).unwrap();
    assert_eq!(&*boxed_str, text);
    let _alloc = Box::allocator(&boxed_str);

    let numbers: &[i32] = &[1, 2, 3, 4, 5];
    let boxed_slice = Box::try_clone_from_ref_in(numbers, System).unwrap();
    assert_eq!(&*boxed_slice, numbers);

    // 3. Feature Gates Verification (Implicitly tested by compiling with #![feature(...)] above)

    // 4. Expanding to Arc and Rc (Bonus Coverage)
    let arc_str = Arc::try_clone_from_ref_in(text, System).unwrap();
    assert_eq!(&*arc_str, text);

    let rc_str = Rc::try_clone_from_ref_in(text, System).unwrap();
    assert_eq!(&*rc_str, text);
}
