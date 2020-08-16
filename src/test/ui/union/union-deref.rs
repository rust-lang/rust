//! Test the part of RFC 2514 that is about not applying `DerefMut` coercions
//! of union fields.
#![feature(untagged_unions)]

use std::mem::ManuallyDrop;

union U<T> { x:(), f: ManuallyDrop<(T,)> }

fn main() {
    let mut u : U<Vec<i32>> = U { x: () };
    unsafe { (*u.f).0 = Vec::new() }; // explicit deref, this compiles
    unsafe { u.f.0 = Vec::new() }; //~ERROR not automatically applying `DerefMut` on union field
}
