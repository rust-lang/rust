//@ ignore-cross-compile
// `ln` is actually `cp` on msys.
//@ ignore-windows

use run_make_support::{rustc, tmp_dir};
use std::process::Command;

fn main() {
    let out = tmp_dir().join("foo.xxx");

    rustc().input("foo.rs").crate_type("rlib").output(&out).run();
    let output = Command::new("ln")
        .arg("-nsf")
        .arg(out)
        .arg(tmp_dir().join("libfoo.rlib"))
        .output()
        .unwrap();
    assert!(output.status.success());
    rustc().input("bar.rs").library_search_path(tmp_dir()).run();
}
