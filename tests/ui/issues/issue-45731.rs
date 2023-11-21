// run-pass
#![allow(unused_variables)]
// compile-flags:--test -g

#[cfg(target_os = "macos")]
#[test]
fn simple_test() {
    use std::{env, panic, fs};

    // Find our dSYM and replace the DWARF binary with an empty file
    let mut dsym_path = env::current_exe().unwrap();
    let executable_name = dsym_path.file_name().unwrap().to_str().unwrap().to_string();
    assert!(dsym_path.pop()); // Pop executable
    dsym_path.push(format!("{}.dSYM/Contents/Resources/DWARF/{0}", executable_name));
    {
        let file = fs::OpenOptions::new().read(false).write(true).truncate(true).create(false)
            .open(&dsym_path).unwrap();
    }

    env::set_var("RUST_BACKTRACE", "1");

    // We don't need to die of panic, just trigger a backtrace
    let _ = panic::catch_unwind(|| {
        assert!(false);
    });
}
