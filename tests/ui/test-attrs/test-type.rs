// compile-flags: --test -Zpanic-abort-tests
// run-flags: --test-threads=1
// check-run-results
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
// ignore-emscripten no threads support
// run-pass

#[test]
fn test_ok() {
    let _a = true;
}

#[test]
#[should_panic]
fn test_panic() {
    panic!();
}

#[test]
#[ignore = "msg"]
fn test_no_run() {
    loop {
        println!("Hello, world");
    }
}

fn main() {}
