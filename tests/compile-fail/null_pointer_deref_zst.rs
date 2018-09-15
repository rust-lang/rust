fn main() {
    let x: () = unsafe { *std::ptr::null() }; //~ ERROR constant evaluation error: invalid use of NULL pointer
    panic!("this should never print: {:?}", x);
}
