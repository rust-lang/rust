//@ edition: 2024
//@ run-pass

trait Super {
    fn foo(&self) -> i32 {
        27
    }
}

trait Sub: Super {}

impl<'a> dyn Sub + 'a {
    fn foo(&self) -> i32 {
        42
    }
}

impl Super for i32 {}
impl Sub for i32 {}

fn main() {
    let x = &0i32;
    assert_eq!(x.foo(), 27);

    let x: &dyn Sub = &0i32;
    assert_eq!(x.foo(), 42);
    assert_eq!(<dyn Sub>::foo(x), 42);
    assert_eq!(<dyn Sub as Super>::foo(x), 27);

    let x: &(dyn Sub + Send) = &0i32;
    assert_eq!(x.foo(), 27);
}
