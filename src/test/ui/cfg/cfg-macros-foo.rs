// run-pass
// compile-flags: --cfg foo

// check that cfg correctly chooses between the macro impls (see also
// cfg-macros-notfoo.rs)


#[cfg(foo)]
#[macro_use]
mod foo {
    macro_rules! bar {
        () => { true }
    }
}

#[cfg(not(foo))]
#[macro_use]
mod foo {
    macro_rules! bar {
        () => { false }
    }
}

pub fn main() {
    assert!(bar!())
}
