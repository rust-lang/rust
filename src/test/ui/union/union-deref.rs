//! Test the part of RFC 2514 that is about not applying `DerefMut` coercions
//! of union fields.
#![feature(untagged_unions)]

use std::mem::ManuallyDrop;

union U1<T> { x:(), f: ManuallyDrop<(T,)> }

union U2<T> { x:(), f: (ManuallyDrop<(T,)>,) }

fn main() {
    let mut u : U1<Vec<i32>> = U1 { x: () };
    unsafe { (*u.f).0 = Vec::new() }; // explicit deref, this compiles
    unsafe { u.f.0 = Vec::new() }; //~ERROR not automatically applying `DerefMut` on union field

    let mut u : U2<Vec<i32>> = U2 { x: () };
    unsafe { (*u.f.0).0 = Vec::new() }; // explicit deref, this compiles
    unsafe { u.f.0.0 = Vec::new() }; //~ERROR not automatically applying `DerefMut` on union field
}
