// run-pass
// compile-flags: --test
#![feature(allow_fail)]
#![feature(cfg_panic)]

#[test]
#[allow_fail]
fn test1() {
    #[cfg(not(panic = "abort"))]
    panic!();
}

#[test]
#[allow_fail]
fn test2() {
    assert!(true);
}
