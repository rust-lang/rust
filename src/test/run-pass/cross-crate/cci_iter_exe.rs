// run-pass
// aux-build:cci_iter_lib.rs

extern crate cci_iter_lib;

pub fn main() {
    //let bt0 = sys::rusti::frame_address(1);
    //println!("%?", bt0);
    cci_iter_lib::iter(&[1, 2, 3], |i| {
        println!("{}", *i);
        //assert_eq!(bt0, sys::rusti::frame_address(2));
    })
}
