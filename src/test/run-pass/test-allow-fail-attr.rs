// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: --test
#![feature(allow_fail)]

#[test]
#[allow_fail]
fn test1() {
    panic!();
}

#[test]
#[allow_fail]
fn test2() {
    assert!(true);
}
