//@ known-bug: #152414
//@ needs-rustc-debug-assertions
#![feature(generic_assert)]
fn main() {
    assert!(size_of(val, 1) >= 1);
}
