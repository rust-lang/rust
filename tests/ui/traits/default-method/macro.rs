//@ run-pass


trait Foo {
    fn bar(&self) -> String {
        format!("test")
    }
}

enum Baz {
    Quux
}

impl Foo for Baz {
}

pub fn main() {
    let q = Baz::Quux;
    assert_eq!(q.bar(), "test".to_string());
}
