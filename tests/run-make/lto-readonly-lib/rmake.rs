// When the compiler is performing link time optimization, it will
// need to copy the original rlib file, set the copy's permissions to read/write,
// and modify that copy - even if the original
// file is read-only. This test creates a read-only rlib, and checks that
// compilation with LTO succeeds.
// See https://github.com/rust-lang/rust/pull/17619

//@ ignore-cross-compile

use run_make_support::fs_wrapper;
use run_make_support::{cwd, run, rustc};

fn main() {
    rustc().input("lib.rs").run();
    let entries = fs_wrapper::read_dir(cwd());
    for entry in entries {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("rlib") {
            let mut perms = fs_wrapper::metadata(entry.path()).permissions();
            perms.set_readonly(true);
            fs_wrapper::set_permissions(entry.path(), perms);
        }
    }
    rustc().input("main.rs").arg("-Clto").run();
    run("main");
}
