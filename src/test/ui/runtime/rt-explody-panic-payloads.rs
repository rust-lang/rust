// run-pass
// needs-unwind
// ignore-emscripten no processes
// ignore-sgx no processes

use std::env;
use std::process::Command;

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        std::panic::panic_any(Bomb);
    }
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let output = match &args[..] {
        [me] => Command::new(&me).arg("plant the").output(),
        [..] => std::panic::panic_any(Bomb),
    }.expect("running the command should have succeeded");
    println!("{:#?}", output);
    let stderr = std::str::from_utf8(&output.stderr).expect("UTF-8 stderr");
    // the drop of the panic payload cannot unwind anymore:
    assert!(stderr.contains("fatal runtime error: drop of the panic payload panicked"));
}
