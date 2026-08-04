// #160439: nested #[doc(hidden)] modules made the visible_parent_map BFS
// exponential. compiletest has no per-test timeout, so time it here.
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use run_make_support::{env_var, rust_lib_name, rustc};

fn main() {
    rustc().input("dep.rs").edition("2021").crate_type("lib").run();

    // the bug made rustdoc hang; kill it if it takes too long
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
