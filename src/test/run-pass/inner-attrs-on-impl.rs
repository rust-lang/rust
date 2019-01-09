struct Foo;

impl Foo {
    #![cfg(cfg_that_surely_doesnt_exist)]

    fn method(&self) -> bool { false }
}

impl Foo {
    #![cfg(not(cfg_that_surely_doesnt_exist))]

    // check that we don't eat attributes too eagerly.
    #[cfg(cfg_that_surely_doesnt_exist)]
    fn method(&self) -> bool { false }

    fn method(&self) -> bool { true }
}


pub fn main() {
    assert!(Foo.method());
}
