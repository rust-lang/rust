// Regression test for https://github.com/rust-lang/rust/issues/160439.
// Nested `#[doc(hidden)]` modules used to make the visible_parent_map BFS
// re-enqueue the same subtree for every parent path, which is exponential.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use run_make_support::{env_var, rust_lib_name, rustc};

fn main() {
    rustc().input("dep.rs").edition("2021").crate_type("lib").run();

    // The bug made rustdoc hang here, so kill it if it takes too long.
    // compiletest has no per-test timeout, hence the manual watchdog.
    let mut child = Command::new(env_var("RUSTDOC"))
        .args(["--edition", "2021", "--crate-name", "main", "-o", "target"])
        .arg(format!("--extern=dep={}", rust_lib_name("dep")))
        .arg("main.rs")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            panic!("rustdoc timed out: visible_parent_map regression?");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    assert!(status.success());
}
