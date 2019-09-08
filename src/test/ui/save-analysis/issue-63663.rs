// check-pass
// compile-flags: -Zsave-analysis

// Check that this doesn't ICE when processing associated const in formal
// argument and return type of functions defined inside function/method scope.

pub trait Trait {
    type Assoc;
}

pub struct A;

pub fn func() {
    fn _inner1<U: Trait>(_: U::Assoc) {}
    fn _inner2<U: Trait>() -> U::Assoc { unimplemented!() }

    impl A {
        fn _inner1<U: Trait>(self, _: U::Assoc) {}
        fn _inner2<U: Trait>(self) -> U::Assoc { unimplemented!() }
    }
}

fn main() {}
