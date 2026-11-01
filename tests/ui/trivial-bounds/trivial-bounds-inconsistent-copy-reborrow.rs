//@ run-pass
#![feature(trivial_bounds)]
#![allow(unused)]

fn reborrow_mut<'a>(t: &'a &'a mut i32) -> &'a mut i32 where &'a mut i32: Copy {
    *t
}

fn copy_reborrow_mut<'a>(t: &'a &'a mut i32) -> &'a mut i32 where &'a mut i32: Copy {
    {*t}
}

fn main() {}
