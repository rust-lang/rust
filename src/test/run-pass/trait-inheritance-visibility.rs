mod traits {
    pub trait Foo { fn f() -> int; }

    impl int: Foo { fn f() -> int { 10 } }
}

trait Quux: traits::Foo { }
impl<T: traits::Foo> T: Quux { }

// Foo is not in scope but because Quux is we can still access
// Foo's methods on a Quux bound typaram
fn f<T: Quux>(x: &T) {
    assert x.f() == 10;
}

fn main() {
    f(&0)
}