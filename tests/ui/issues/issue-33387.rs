//@ run-pass
#![feature(rustc_attrs)]

use std::sync::Arc;

trait Foo {
    fn get(&self) -> [u8; 2];
}

impl Foo for [u8; 2] {
    fn get(&self) -> [u8; 2] {
        *self
    }
}

struct Bar<T: ?Sized>(T);

fn unsize_fat_ptr<'a>(x: &'a Bar<dyn Foo + Send + 'a>) -> &'a Bar<dyn Foo + 'a> {
    x
}

fn unsize_nested_fat_ptr(x: Arc<dyn Foo + Send>) -> Arc<dyn Foo> {
    x
}

fn main() {
    let x: Box<Bar<dyn Foo + Send>> = Box::new(Bar([1,2]));
    assert_eq!(unsize_fat_ptr(&*x).0.get(), [1, 2]);

    let x: Arc<dyn Foo + Send> = Arc::new([3, 4]);
    assert_eq!(unsize_nested_fat_ptr(x).get(), [3, 4]);
}
