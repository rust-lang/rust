//@ only-windows

// Ensure that we aren't relying on any non-system DLLs when compiling and running
// a "hello world" application by setting `PATH` to `C:\Windows\System32`.

use run_make_support::{run, rustc};
use std::env;
use std::path::PathBuf;

fn main() {
    let windows_dir = env::var("SystemRoot").unwrap();
    let system32: PathBuf = [&windows_dir, "System32"].iter().collect();
    rustc().input("hello.rs").env("PATH", system32).run();
    run("hello");
}
