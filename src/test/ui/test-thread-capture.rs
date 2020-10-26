// compile-flags: --test
// run-fail
// run-flags: --test-threads=1
// check-run-results
// exec-env:RUST_BACKTRACE=0
// ignore-emscripten no threads support

#[test]
fn thready_pass() {
    println!("fee");
    std::thread::spawn(|| {
        println!("fie");
        println!("foe");
    })
    .join()
    .unwrap();
    println!("fum");
}

#[test]
fn thready_fail() {
    println!("fee");
    std::thread::spawn(|| {
        println!("fie");
        println!("foe");
    })
    .join()
    .unwrap();
    println!("fum");
    panic!();
}
