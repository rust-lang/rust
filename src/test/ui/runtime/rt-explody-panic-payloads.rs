// run-pass
// needs-unwind
// ignore-emscripten no processes
// ignore-sgx no processes
// ignore-wasm32-bare no unwinding panic
// ignore-avr no unwinding panic
// ignore-nvptx64 no unwinding panic

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
    }
    .expect("running the command should have succeeded");
    println!("{:#?}", output);
    let stderr = std::str::from_utf8(&output.stderr);

    // The standard library used in tests is built with -Z panic-in-drop=abort
    // which causes the unwind to fail immediately since the catch_unwind in
    // the standard library has been optimized away as unreachable. Even if it
    // weren't optimized away the unwind would still be forcibly stopped in
    // drop_in_place which cannot unwind.
    //
    // When the standard library is built with -Z panic-in-drop=unwind this
    // fails with a specific error message: "drop of the panic payload panicked"
    assert!(
        stderr
            .map(|v| { v.contains("fatal runtime error: failed to initiate panic") })
            .unwrap_or(false)
    );
}
