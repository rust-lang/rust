// nested function calls with cast.
fn main() {
    self.ptr
        .set(intrinsics::arith_offset(self.ptr.get() as *mut u8, 1) as *mut T);
    self.ptr
        .set(intrinsics::arith_offset(self.ptr.get(), mem::size_of::<T>() as isize) as *mut u8);
}
