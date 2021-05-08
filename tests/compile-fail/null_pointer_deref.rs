#[allow(deref_nullptr)]
fn main() {
    let x: i32 = unsafe { *std::ptr::null() }; //~ ERROR null pointer is not a valid pointer for this operation
    panic!("this should never print: {}", x);
}
