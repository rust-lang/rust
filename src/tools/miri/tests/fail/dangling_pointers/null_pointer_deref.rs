#[allow(deref_nullptr)]
fn main() {
    let x: i32 = unsafe { *std::ptr::null() }; //~ ERROR: null pointer is a dangling pointer
    panic!("this should never print: {}", x);
}
