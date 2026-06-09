//! Regression test for <https://github.com/rust-lang/rust/issues/146174>.
//! Ensure that we don't ICE when an associated function returns an associated type.

mod m {
    pub trait Project {
        type Assoc;
    }
    pub struct Foo {
        _priv: (),
    }
    impl Foo {
        fn new<T: Project>() -> T::Assoc {
            todo!()
        }
    }
}
fn main() {
    let _ = m::Foo {}; //~ ERROR: cannot construct `Foo`
}
