// This is a regression test for https://github.com/rust-lang/rust/issues/118047
//@ build-pass
//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+DataflowConstProp

#![crate_type = "lib"]

pub struct State {
    inner: bool
}

pub fn make() -> State {
    State {
        inner: true
    }
}
