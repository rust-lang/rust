#![allow(unused_variables)]
// no-prefer-dynamic
// ignore-cross-compile

use std::env;
use std::fs;
use std::process;
use std::str;
use std::path::PathBuf;

fn main() {
    // If we're the child, make sure we were invoked correctly
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        // FIXME: This should check the whole `args[0]` instead of just
        // checking that it ends_with the executable name. This
        // is needed because of Windows, which has a different behavior.
        // See #15149 for more info.
        return assert!(args[0].ends_with(&format!("mytest{}",
                                                  env::consts::EXE_SUFFIX)));
    }

    test();
}

fn test() {
    // If we're the parent, copy our own binary to a new directory.
    let my_path = env::current_exe().unwrap();
    let my_dir  = my_path.parent().unwrap();

    let child_dir = PathBuf::from(env::var_os("RUST_TEST_TMPDIR").unwrap());
    let child_dir = child_dir.join("issue-15140-child");
    fs::create_dir_all(&child_dir).unwrap();

    let child_path = child_dir.join(&format!("mytest{}",
                                             env::consts::EXE_SUFFIX));
    fs::copy(&my_path, &child_path).unwrap();

    // Append the new directory to our own PATH.
    let path = {
        let mut paths: Vec<_> = env::split_paths(&env::var_os("PATH").unwrap()).collect();
        paths.push(child_dir.to_path_buf());
        env::join_paths(paths).unwrap()
    };

    let child_output = process::Command::new("mytest").env("PATH", &path)
                                                      .arg("child")
                                                      .output().unwrap();

    assert!(child_output.status.success(),
            format!("child assertion failed\n child stdout:\n {}\n child stderr:\n {}",
                    str::from_utf8(&child_output.stdout).unwrap(),
                    str::from_utf8(&child_output.stderr).unwrap()));
}
