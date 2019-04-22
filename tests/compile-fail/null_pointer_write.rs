fn main() {
    unsafe { *std::ptr::null_mut() = 0i32 }; //~ ERROR invalid use of NULL pointer
}
