#[link(name = "foo", kind = "static")]
extern {
    fn test_start(f: extern fn());
    fn test_end();
}

fn main() {
    unsafe {
        test_start(test_middle);
    }
}

struct A;

impl Drop for A {
    fn drop(&mut self) {
    }
}

extern fn test_middle() {
    let _a = A;
    foo();
}

fn foo() {
    let _a = A;
    unsafe {
        test_end();
    }
}
