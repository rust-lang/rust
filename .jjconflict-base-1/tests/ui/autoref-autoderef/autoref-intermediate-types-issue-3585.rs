//@ run-pass

trait Foo {
    fn foo(&self) -> String;
}

impl<T:Foo> Foo for Box<T> {
    fn foo(&self) -> String {
        format!("box {}", (**self).foo())
    }
}

impl Foo for usize {
    fn foo(&self) -> String {
        format!("{}", *self)
    }
}

pub fn main() {
    let x: Box<_> = Box::new(3);
    assert_eq!(x.foo(), "box 3".to_string());
}
