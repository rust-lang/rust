fn main() {
    let x: i32 = unsafe { *std::ptr::null() }; //~ ERROR inbounds test failed: 0x0 is not a valid pointer
    panic!("this should never print: {}", x);
}
