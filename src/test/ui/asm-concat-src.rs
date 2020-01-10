// run-pass
// pretty-expanded FIXME #23616

#![feature(asm)]

pub fn main() {
    unsafe { asm!(concat!("", "")) };
}
