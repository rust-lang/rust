//@ run-pass

struct Foo;

impl Foo {
    #![cfg(false)]

    fn method(&self) -> bool { false }
}

impl Foo {
    #![cfg(not(FALSE))]

    // check that we don't eat attributes too eagerly.
    #[cfg(false)]
    fn method(&self) -> bool { false }

    fn method(&self) -> bool { true }
}


pub fn main() {
    assert!(Foo.method());
}
