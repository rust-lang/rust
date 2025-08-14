//@ check-pass
mod foo {
    struct Priv;
    mod bar {
        use crate::foo::Priv;
        pub(super) fn f(_: Priv) {}
        pub(crate) fn g(_: Priv) {}
        pub(crate) fn h(_: Priv) {}
    }
}

fn main() { }
