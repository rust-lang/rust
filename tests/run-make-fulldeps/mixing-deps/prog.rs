extern crate dylib;
extern crate both;

use std::mem;

fn main() {
    assert_eq!(unsafe { mem::transmute::<&isize, usize>(&both::foo) },
               dylib::addr());
}
