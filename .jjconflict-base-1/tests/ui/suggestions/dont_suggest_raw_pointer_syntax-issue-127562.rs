fn main() {
    let val = 2;
    let ptr = std::ptr::addr_of!(val);
    unsafe {
        *ptr = 3; //~ ERROR cannot assign to `*ptr`, which is behind a `*const` pointer
    }
}
