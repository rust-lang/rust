// run-pass
// pretty-expanded FIXME #23616

#![allow(drop_copy)]

fn main() {
    use ::std::mem;
    mem::drop(2_usize);
}
