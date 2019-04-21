fn main() {
    let x: i32 = unsafe { *std::ptr::null() }; //~ ERROR invalid use of NULL pointer
    panic!("this should never print: {}", x);
}
