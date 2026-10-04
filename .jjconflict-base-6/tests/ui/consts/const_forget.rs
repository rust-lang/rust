//@ check-pass

#![allow(forgetting_copy_types)]

use std::mem::forget;

const _: () = forget(0i32);
const _: () = forget(Vec::<Vec<Box<i32>>>::new());

// Writing this function signature without const-forget
// triggers compiler errors:
// 1) That we use a non-const fn inside a const fn
// 2) without the forget, it complains about the destructor of Box
//
// FIXME: this method cannot be called in const-eval yet, as Box isn't
// const constructable
#[allow(unused)]
const fn const_forget_box<T: ?Sized>(b: Box<T>) {
    forget(b);
}

fn main() {}
