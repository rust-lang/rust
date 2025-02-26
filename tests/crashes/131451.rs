//@ known-bug: #131451
//@ needs-rustc-debug-assertions
//@ compile-flags: -Zmir-enable-passes=+GVN -Zmir-enable-passes=+JumpThreading --crate-type=lib

pub fn fun(terminate: bool) {
    while true {}

    while !terminate {}
}
