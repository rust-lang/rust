#![cfg(windows)]

use std::fs::File;
use std::io::IsTerminal;

unsafe extern "system" {
    fn AllocConsole() -> i32;
}

fn ensure_open(path: &str, read: bool, write: bool) -> File {
    let try_open = || {
        let mut opts = File::options();
        if read {
            opts.read(true);
        }
        if write {
            opts.write(true);
        }
        opts.open(path)
    };

    try_open().unwrap_or_else(|_| {
        unsafe {
            AllocConsole();
        }
        try_open().unwrap_or_else(|e| {
            panic!("{path} unavailable even after AllocConsole: {e}");
        })
    })
}

#[test]
fn write_only_conout_is_terminal() {
    assert!(ensure_open(r"\\.\CONOUT$", false, true).is_terminal());
}

#[test]
fn readwrite_conout_is_terminal() {
    assert!(ensure_open(r"\\.\CONOUT$", true, true).is_terminal());
}

#[test]
fn conin_is_terminal() {
    assert!(ensure_open(r"\\.\CONIN$", true, false).is_terminal());
}

#[test]
fn nul_is_not_terminal() {
    let file = File::options().read(true).write(true).open(r"\\.\NUL").unwrap();
    assert!(!file.is_terminal());
}

#[test]
fn regular_file_is_not_terminal() {
    let path = std::env::temp_dir().join("__rust_test_is_terminal.tmp");
    let file = File::create(&path).unwrap();
    assert!(!file.is_terminal());
    drop(file);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn std_handles_no_panic() {
    let _ = std::io::stdout().is_terminal();
    let _ = std::io::stderr().is_terminal();
    let _ = std::io::stdin().is_terminal();
}
