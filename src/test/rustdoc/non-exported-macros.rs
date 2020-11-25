// compile-flags: --document-private-items

fn foo_fn() {}

pub fn pub_foo_fn() {}

macro_rules! foo_macro {
    () => { };
}

#[macro_export]
macro_rules! exported_foo_macro {
    () => { };
}

// TODO: add `@has` checks
