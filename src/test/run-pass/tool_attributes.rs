// Scoped attributes should not trigger an unused attributes lint.

#![deny(unused_attributes)]

fn main() {
    #[rustfmt::skip]
    foo ();
}

fn foo() {
    assert!(true);
}
