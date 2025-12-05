//@ run-pass
//@ compile-flags: --test
//@ run-flags: --test-threads=1 test1 test2
//@ check-run-results
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ needs-threads

#[test]
fn test1() {}

#[test]
fn test2() {}

#[test]
fn test3() {
    panic!("this should not run");
}
