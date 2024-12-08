//@ run-pass

#![allow(dropping_copy_types)]

fn main() {
    use ::std::mem;
    mem::drop(2_usize);
}
