#![cfg(not(target_env = "sgx"))]

use std::{env, fs, process, str};

mod common;

#[test]
// Process spawning not supported by Miri, Emscripten and wasi
#[cfg_attr(any(miri, target_os = "emscripten", target_os = "wasi"), ignore)]
fn issue_15149() {
    // If we're the parent, copy our own binary to a new directory.
    let my_path = env::current_exe().unwrap();

    let temp = common::tmpdir();
    let child_dir = temp.join("issue-15140-child");
    fs::create_dir_all(&child_dir).unwrap();

    let child_path = child_dir.join(&format!("mytest{}", env::consts::EXE_SUFFIX));
    fs::copy(&my_path, &child_path).unwrap();

    // Append the new directory to our own PATH.
    let path = {
        let mut paths: Vec<_> = env::split_paths(&env::var_os("PATH").unwrap()).collect();
        paths.push(child_dir.to_path_buf());
        env::join_paths(paths).unwrap()
    };

    let child_output =
        process::Command::new("mytest").env("PATH", &path).arg("child").output().unwrap();

    assert!(
        child_output.status.success(),
        "child assertion failed\n child stdout:\n {}\n child stderr:\n {}",
        str::from_utf8(&child_output.stdout).unwrap(),
        str::from_utf8(&child_output.stderr).unwrap()
    );
}
