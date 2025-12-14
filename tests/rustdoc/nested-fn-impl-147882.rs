//@ compile-flags: -Zunstable-options --generate-link-to-definition

// See: https://github.com/rust-lang/rust/issues/147882

#![crate_name = "foo"]

trait Fun {
    fn fun();
}

trait Other {}

impl Fun for () {
    fn fun() {
        impl<E> Other for E
        where
            E: std::str::FromStr,
            E::Err: Send,
        {
        }
    }
}
