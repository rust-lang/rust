// ignore-emscripten
// ignore-wasm32
// ignore-sgx no processes

use std::env;
use std::process::Command;

fn main() {
    if env::args().nth(1).map(|s| s == "print").unwrap_or(false) {
        for (k, v) in env::vars() {
            println!("{}={}", k, v);
        }
        return
    }

    let me = env::current_exe().unwrap();
    let result = Command::new(me).arg("print").output().unwrap();
    let output = String::from_utf8(result.stdout).unwrap();

    for (k, v) in env::vars() {
        assert!(output.contains(&format!("{}={}", k, v)),
                "output doesn't contain `{}={}`\n{}",
                k, v, output);
    }
}
