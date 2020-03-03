// run-pass
// compile-flags: -Z chalk

trait Foo { }

impl Foo for i32 { }

struct S<T: Foo> {
    x: T,
}

fn only_foo<T: Foo>(_x: &T) { }

impl<T> S<T> {
    // Test that we have the correct environment inside an inherent method.
    fn dummy_foo(&self) {
        only_foo(&self.x)
    }
}

fn main() {
    let s = S {
        x: 5,
    };

    s.dummy_foo();
}
