fn f(p: *mut u8) {
    *p = 0; //~ ERROR dereference of raw pointer is unsafe
    return;
}

fn main() {
}
