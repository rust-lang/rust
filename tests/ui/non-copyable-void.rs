use std::ffi::c_void;

fn main() {
    let x : *const Vec<isize> = &vec![1,2,3];
    let y : *const c_void = x as *const c_void;
    unsafe {
        let _z = (*y).clone();
        //~^ ERROR no method named `clone` found
    }
}
