// Check that reborrows are still illegal with Copy mutable references
#![feature(trivial_bounds)]
#![allow(unused)]

fn reborrow_mut<'a>(t: &'a &'a mut i32) -> &'a mut i32 where &'a mut i32: Copy {
    *t //~ ERROR
}

fn copy_reborrow_mut<'a>(t: &'a &'a mut i32) -> &'a mut i32 where &'a mut i32: Copy {
    {*t} //~ ERROR
}

fn main() {}
