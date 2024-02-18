//@ run-rustfix

#![feature(box_patterns)]
#![allow(dead_code)]

fn foo(f: Option<Box<i32>>) {
    match f {
        Some(ref box _i) => {},
        //~^ ERROR switch the order of `ref` and `box`
        None => {}
    }
}

fn main() { }
