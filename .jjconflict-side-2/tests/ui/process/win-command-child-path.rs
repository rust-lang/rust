//@ run-pass
//@ only-windows
//@ needs-subprocess
//@ no-prefer-dynamic

// Test Windows std::process::Command search path semantics when setting PATH on the child process.
// NOTE: The exact semantics here are (possibly) subject to change.

use std::process::Command;
use std::{env, fs, path};

fn main() {
    if env::args().skip(1).any(|s| s == "--child") {
        child();
    } else if env::args().skip(1).any(|s| s == "--parent") {
        parent();
    } else {
        setup();
    }
}

// Set up the directories so that there are three app dirs:
// app: Where the parent app is run from
// parent: In the parent's PATH env var
// child: In the child's PATH env var
fn setup() {
    let exe = env::current_exe().unwrap();

    fs::create_dir_all("app").unwrap();
    fs::copy(&exe, "app/myapp.exe").unwrap();
    fs::create_dir_all("parent").unwrap();
    fs::copy(&exe, "parent/myapp.exe").unwrap();
    fs::create_dir_all("child").unwrap();
    fs::copy(&exe, "child/myapp.exe").unwrap();

    let parent_path = path::absolute("parent").unwrap();
    let status =
        Command::new("./app/myapp.exe").env("PATH", parent_path).arg("--parent").status().unwrap();
    // print the status in case of abnormal exit
    dbg!(status);
    assert!(status.success());
}

// The child simply prints the name of its parent directory.
fn child() {
    let exe = env::current_exe().unwrap();
    let parent = exe.parent().unwrap().file_name().unwrap();
    println!("{}", parent.display());
}

fn parent() {
    let exe = env::current_exe().unwrap();
    let name = exe.file_name().unwrap();

    // By default, the application dir will be search first for the exe
    let output = Command::new(&name).arg("--child").output().unwrap();
    assert_eq!(output.stdout, b"app\n");

    // Setting an environment variable should not change the above.
    let output = Command::new(&name).arg("--child").env("a", "b").output().unwrap();
    assert_eq!(output.stdout, b"app\n");

    // Setting a child path means that path will be searched first.
    let child_path = path::absolute("child").unwrap();
    let output = Command::new(&name).arg("--child").env("PATH", child_path).output().unwrap();
    assert_eq!(output.stdout, b"child\n");

    // Setting a child path that does not contain the exe (currently) means
    // we fallback to searching the app dir.
    let output = Command::new(&name).arg("--child").env("PATH", "").output().unwrap();
    assert_eq!(output.stdout, b"app\n");
}
