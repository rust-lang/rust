//ignore-test

// This file is used by the ice-4727 test but isn't itself a test.
//
pub trait Trait {
    fn fun(par: &str) -> &str;
}

impl Trait for str {
    fn fun(par: &str) -> &str {
        &par[0..1]
    }
}
