// xfail-test
enum x = ();
impl x {
    unsafe fn with() { } // This should fail
}

fn main() {
    x(()).with();
}
