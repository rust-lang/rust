// compile-flags: -Cmetadata=aux

pub struct Foo;

impl Foo {
    pub fn new<F>(f: F) -> Foo where F: FnMut() -> i32 {
        loop {}
    }
}
