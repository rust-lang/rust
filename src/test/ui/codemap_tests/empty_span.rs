#![feature(optin_builtin_traits)]
fn main() {
    struct Foo;

    impl !Sync for Foo {}

    unsafe impl Send for &'static Foo { } //~ ERROR cross-crate traits with a default impl
}
