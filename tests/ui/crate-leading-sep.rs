//@ run-pass
//@ pretty-expanded FIXME #23616

#![allow(dropping_copy_types)]

fn main() {
    use ::std::mem;
    mem::drop(2_usize);
}
