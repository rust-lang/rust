// Regression test for <https://github.com/rust-lang/rust/issues/147855>
//@ edition:2024
//@ run-rustfix
//@ rustfix-only-machine-applicable
//@ check-pass

mod m {
    macro_rules! define_new_macro {
        ($name:ident) => {
            macro_rules! $name {
                () => {};
            }
            pub(crate) use $name;
        };
    }

    define_new_macro!(item_used);
    define_new_macro!(item_unused);
}

fn main() {
    m::item_used!();
}
