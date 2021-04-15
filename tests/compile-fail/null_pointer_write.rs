#[allow(deref_nullptr)]
fn main() {
    unsafe { *std::ptr::null_mut() = 0i32 }; //~ ERROR inbounds test failed: 0x0 is not a valid pointer
}
