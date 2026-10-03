fn main() {
    fn f() {}

    let g = f as fn() as unsafe fn();
    unsafe {
        g();
    }
}
