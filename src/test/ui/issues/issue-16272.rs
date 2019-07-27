// run-pass
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

use std::process::Command;
use std::env;

fn main() {
    let len = env::args().len();

    if len == 1 {
        test();
    } else {
        assert_eq!(len, 3);
    }
}

fn test() {
    let status = Command::new(&env::current_exe().unwrap())
                         .arg("foo").arg("")
                         .status().unwrap();
    assert!(status.success());
}
