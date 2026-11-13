// The `#[expect]` sharing with derive-generated code also applies when the
// derive itself is expanded from a macros 2.0 macro: the impl is still created
// by the innermost (derive) expansion, so it is recognized as derive-generated
// no matter what macro produced the derived item. A genuinely unfulfilled
// expectation is still reported exactly once.

//@ check-pass

#![feature(decl_macro)]

macro fulfilled() {
    #[derive(Debug)]
    #[expect(non_upper_case_globals)]
    pub struct Wrapper<const n: usize> {
        pub field: [u8; n],
    }
}

fulfilled!();

macro passthrough($i:item) {
    $i
}

passthrough! {
    #[derive(Debug)]
    #[expect(non_upper_case_globals)]
    pub struct WrapperPassthrough<const n: usize> {
        pub field: [u8; n],
    }
}

macro unfulfilled() {
    #[derive(Debug)]
    #[expect(unexpected_cfgs)]
    //~^ WARN this lint expectation is unfulfilled
    pub struct Unfulfilled {
        pub x: i64,
        #[cfg(false)]
        pub gone: u8,
    }
}

unfulfilled!();

fn main() {}
