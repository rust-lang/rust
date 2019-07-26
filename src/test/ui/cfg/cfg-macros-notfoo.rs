// run-pass
// compile-flags:

// check that cfg correctly chooses between the macro impls (see also
// cfg-macros-foo.rs)


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
    assert!(!bar!())
}
