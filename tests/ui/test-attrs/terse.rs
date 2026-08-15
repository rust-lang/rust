//@ compile-flags: --test
//@ run-fail
//@ run-flags: --test-threads=1 --quiet
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ needs-threads
//@ needs-unwind

#[test]
fn abc() {
    panic!();
}

#[test]
fn foo() {
    panic!();
}

#[test]
fn foo2() {
    panic!();
}

// run a whole bunch of tests so we can see what happens when we go over 88 columns
#[test] fn f0() {}
#[test] fn f1() {}
#[test] fn f2() {}
#[test] fn f3() {}
#[test] fn f4() {}
#[test] fn f5() {}
#[test] fn f6() {}
#[test] fn f7() {}
#[test] fn f8() {}
#[test] fn f9() {}
#[test] fn f10() {}
#[test] fn f11() {}
#[test] fn f12() {}
#[test] fn f13() {}
#[test] fn f14() {}
#[test] fn f15() {}
#[test] fn f16() {}
#[test] fn f17() {}
#[test] fn f18() {}
#[test] fn f19() {}
#[test] fn f20() {}
#[test] fn f21() {}
#[test] fn f22() {}
#[test] fn f23() {}
#[test] fn f24() {}
#[test] fn f25() {}
#[test] fn f26() {}
#[test] fn f27() {}
#[test] fn f28() {}
#[test] fn f29() {}
#[test] fn f30() {}
#[test] fn f31() {}
#[test] fn f32() {}
#[test] fn f33() {}
#[test] fn f34() {}
#[test] fn f35() {}
#[test] fn f36() {}
#[test] fn f37() {}
#[test] fn f38() {}
#[test] fn f39() {}
#[test] fn f40() {}
#[test] fn f41() {}
#[test] fn f42() {}
#[test] fn f43() {}
#[test] fn f44() {}
#[test] fn f45() {}
#[test] fn f46() {}
#[test] fn f47() {}
#[test] fn f48() {}
#[test] fn f49() {}
#[test] fn f50() {}
#[test] fn f51() {}
#[test] fn f52() {}
#[test] fn f53() {}
#[test] fn f54() {}
#[test] fn f55() {}
#[test] fn f56() {}
#[test] fn f57() {}
#[test] fn f58() {}
#[test] fn f59() {}
#[test] fn f60() {}
#[test] fn f61() {}
#[test] fn f62() {}
#[test] fn f63() {}
#[test] fn f64() {}
#[test] fn f65() {}
#[test] fn f66() {}
#[test] fn f67() {}
#[test] fn f68() {}
#[test] fn f69() {}
#[test] fn f70() {}
#[test] fn f71() {}
#[test] fn f72() {}
#[test] fn f73() {}
#[test] fn f74() {}
#[test] fn f75() {}
#[test] fn f76() {}
#[test] fn f77() {}
#[test] fn f78() {}
#[test] fn f79() {}
#[test] fn f80() {}
#[test] fn f81() {}
#[test] fn f82() {}
#[test] fn f83() {}
#[test] fn f84() {}
#[test] fn f85() {}
#[test] fn f86() {}
#[test] fn f87() {}
#[test] fn f88() {}
#[test] fn f89() {}
#[test] fn f90() {}
#[test] fn f91() {}
#[test] fn f92() {}
#[test] fn f93() {}
#[test] fn f94() {}
#[test] fn f95() {}
#[test] fn f96() {}
#[test] fn f97() {}
#[test] fn f98() {}
#[test] fn f99() {}
