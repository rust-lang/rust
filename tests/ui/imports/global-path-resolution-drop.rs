//! Checks global path resolution of `mem::drop` using a leading `::`.

//@ run-pass

#![allow(dropping_copy_types)]

fn main() {
    use ::std::mem;
    mem::drop(2_usize);
}
