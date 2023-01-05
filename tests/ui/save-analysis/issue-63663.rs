// check-pass
// compile-flags: -Zsave-analysis

pub trait Trait {
    type Assoc;
}

pub struct A;

trait Generic<T> {}
impl<T> Generic<T> for () {}

// Don't ICE when resolving type paths in return type `impl Trait`
fn assoc_in_opaque_type_bounds<U: Trait>() -> impl Generic<U::Assoc> {}

// Check that this doesn't ICE when processing associated const in formal
// argument and return type of functions defined inside function/method scope.
pub fn func() {
    fn _inner1<U: Trait>(_: U::Assoc) {}
    fn _inner2<U: Trait>() -> U::Assoc { unimplemented!() }

    impl A {
        fn _inner1<U: Trait>(self, _: U::Assoc) {}
        fn _inner2<U: Trait>(self) -> U::Assoc { unimplemented!() }
    }
}

fn main() {}
