// check-pass

#![feature(offset_of)]
#![warn(unused)]

fn main() {
    core::mem::offset_of!((String,), 0);
    //~^ WARN unused return value of `must_use` that must be used
}
