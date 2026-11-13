//@ check-pass
// Check tautalogically false `Copy` bounds

#![feature(trivial_bounds)]
#![allow(dropping_references, dropping_copy_types)]

fn copy_string(t: String) -> String where String: Copy { //~ WARNING trivial_bounds
    is_copy(&t);
    let x = t;
    drop(t);
    t
}

fn copy_out_string(t: &String) -> String where String: Copy { //~ WARNING trivial_bounds
    *t
}

fn copy_string_with_param<T>(x: String) where String: Copy { //~ WARNING trivial_bounds
    let y = x;
    let z = x;
}

// Check that no reborrowing occurs
fn copy_mut<'a>(t: &&'a mut i32) -> &'a mut i32 where for<'b> &'b mut i32: Copy {
    //~^ WARNING trivial_bounds
    is_copy(t);
    let x = *t;
    drop(x);
    x
}

fn is_copy<T: Copy>(t: &T) {}


fn main() {}
