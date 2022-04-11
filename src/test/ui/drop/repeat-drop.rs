// run-pass

static mut COUNT: usize = 0;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        unsafe {
            COUNT += 1;
        }
    }
}

fn a() {
    let foo = Foo;
    let v: [Foo; 0] = [foo; 0];
    unsafe { assert_eq!(COUNT, 1) }
    std::mem::drop(v);
}

fn b() {
    let foo = Foo;
    let v: [Foo; 1] = [foo; 1];
    unsafe { assert_eq!(COUNT, 0) }
    std::mem::drop(v);
    unsafe { assert_eq!(COUNT, 1) }
}

fn main() {
    a();
    unsafe {
        COUNT = 0;
    }
    b();
}
