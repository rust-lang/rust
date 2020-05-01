fn main() {
    let x: () = unsafe { *std::ptr::null() }; //~ ERROR memory access failed: 0x0 is not a valid pointer
    panic!("this should never print: {:?}", x);
}
