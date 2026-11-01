struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        unsafe {
            FOO = true;
        }
    }
}

static mut FOO: bool = false;

enum Bar {
    A(#[allow(dead_code)] Box<i32>),
    B(Foo),
}

fn main() {
    assert!(unsafe { !FOO });
    drop(Bar::A(Box::new(42)));
    assert!(unsafe { !FOO });
    drop(Bar::B(Foo));
    assert!(unsafe { FOO });
}
