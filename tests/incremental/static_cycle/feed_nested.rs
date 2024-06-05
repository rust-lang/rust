//@ revisions:rpass1 rpass2

//! Test that the following order of instructions will not create duplicate
//! `DefId`s for the nested static items.
// ensure(eval_static_initializer(FOO))
// -> try_mark_green(eval_static_initializer(FOO))
// -> green
// -> replay side effects
// -> create some definitions.
//
// get(eval_static_initializer(FOO))
// -> graph in place
// -> replay
// -> eval_static_initializer.compute
// -> how do we skip re-creating the same definitions ?

#![feature(const_mut_refs)]
#![cfg_attr(rpass2, warn(dead_code))]

pub static mut FOO: &mut i32 = &mut 42;

pub static mut BAR: &mut i32 = unsafe { FOO };

fn main() {
    unsafe {
        assert_eq!(BAR as *mut i32, FOO as *mut i32);
    }
}
