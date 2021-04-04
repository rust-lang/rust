// run-pass

use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

#[cfg(unix)]
fn switch_stdout_to(file: File) {
    use std::os::unix::prelude::*;

    extern "C" {
        fn dup2(old: i32, new: i32) -> i32;
    }

    unsafe {
        assert_eq!(dup2(file.as_raw_fd(), 1), 1);
    }
}

#[cfg(windows)]
fn switch_stdout_to(file: File) {
    use std::os::windows::prelude::*;

    extern "system" {
        fn SetStdHandle(nStdHandle: u32, handle: *mut u8) -> i32;
    }

    const STD_OUTPUT_HANDLE: u32 = (-11i32) as u32;

    unsafe {
        let rc = SetStdHandle(STD_OUTPUT_HANDLE, file.into_raw_handle() as *mut _);
        assert!(rc != 0);
    }
}

fn main() {
    let path = PathBuf::from(env::var_os("RUST_TEST_TMPDIR").unwrap());
    let path = path.join("switch-stdout-output");
    let f = File::create(&path).unwrap();

    println!("foo");
    std::io::stdout().flush().unwrap();
    switch_stdout_to(f);
    println!("bar");
    std::io::stdout().flush().unwrap();

    let mut contents = String::new();
    File::open(&path).unwrap().read_to_string(&mut contents).unwrap();
    assert_eq!(contents, "bar\n");
}
