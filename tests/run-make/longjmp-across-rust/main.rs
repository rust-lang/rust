#[link(name = "foo", kind = "static")]
extern "C" {
    fn test_start(f: extern "C" fn());
    fn test_end();
}

fn main() {
    unsafe {
        test_start(test_middle);
    }
}

extern "C" fn test_middle() {
    foo();
}

fn foo() {
    unsafe {
        test_end();
    }
}
