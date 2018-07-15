fn main() {
    let x: i32 = unsafe { *std::ptr::null() }; //~ ERROR constant evaluation error
    //~^ NOTE invalid use of NULL pointer
    panic!("this should never print: {}", x);
}
