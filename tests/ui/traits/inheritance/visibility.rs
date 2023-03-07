// run-pass

mod traits {
    pub trait Foo { fn f(&self) -> isize; }

    impl Foo for isize { fn f(&self) -> isize { 10 } }
}

trait Quux: traits::Foo { }
impl<T:traits::Foo> Quux for T { }

// Foo is not in scope but because Quux is we can still access
// Foo's methods on a Quux bound typaram
fn f<T:Quux>(x: &T) {
    assert_eq!(x.f(), 10);
}

pub fn main() {
    f(&0)
}
