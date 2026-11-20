extern "C" {
    fn bar();
    fn baz();
}

fn main() {
    unsafe {
        bar();
        baz();
    }
}
