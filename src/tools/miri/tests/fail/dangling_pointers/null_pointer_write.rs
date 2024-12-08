#[allow(deref_nullptr)]
fn main() {
    unsafe { *std::ptr::null_mut() = 0i32 }; //~ ERROR: null pointer
}
