// run-pass

trait Foo: Fn(i32) -> i32 + Send {}

impl<T: ?Sized + Fn(i32) -> i32 + Send> Foo for T {}

fn wants_foo(f: Box<dyn Foo>) -> i32 {
    f(42)
}

fn main() {
    let f = Box::new(|x| x);
    assert_eq!(wants_foo(f), 42);
}
