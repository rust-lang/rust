#![feature(crate_visibility_modifier)]

mod foo {
    struct Priv;
    mod bar {
        use foo::Priv;
        pub(super) fn f(_: Priv) {}
        pub(crate) fn g(_: Priv) {} //~ ERROR E0446
        crate fn h(_: Priv) {} //~ ERROR E0446
    }
}

fn main() { }
