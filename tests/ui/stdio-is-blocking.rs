// run-pass
// ignore-emscripten no processes
// ignore-sgx no processes

use std::env;
use std::io::prelude::*;
use std::process::Command;
use std::thread;

const THREADS: usize = 20;
const WRITES: usize = 100;
const WRITE_SIZE: usize = 1024 * 32;

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 1 {
        parent();
    } else {
        child();
    }
}

fn parent() {
    let me = env::current_exe().unwrap();
    let mut cmd = Command::new(me);
    cmd.arg("run-the-test");
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    assert_eq!(output.stderr.len(), 0);
    assert_eq!(output.stdout.len(), WRITES * THREADS * WRITE_SIZE);
    for byte in output.stdout.iter() {
        assert_eq!(*byte, b'a');
    }
}

fn child() {
    let threads = (0..THREADS).map(|_| {
        thread::spawn(|| {
            let buf = [b'a'; WRITE_SIZE];
            for _ in 0..WRITES {
                write_all(&buf);
            }
        })
    }).collect::<Vec<_>>();

    for thread in threads {
        thread.join().unwrap();
    }
}

#[cfg(unix)]
fn write_all(buf: &[u8]) {
    use std::fs::File;
    use std::mem;
    use std::os::unix::prelude::*;

    let mut file = unsafe { File::from_raw_fd(1) };
    let res = file.write_all(buf);
    mem::forget(file);
    res.unwrap();
}

#[cfg(windows)]
fn write_all(buf: &[u8]) {
    use std::fs::File;
    use std::mem;
    use std::os::windows::raw::*;
    use std::os::windows::prelude::*;

    const STD_OUTPUT_HANDLE: u32 = (-11i32) as u32;

    extern "system" {
        fn GetStdHandle(handle: u32) -> HANDLE;
    }

    let mut file = unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        assert!(!handle.is_null());
        File::from_raw_handle(handle)
    };
    let res = file.write_all(buf);
    mem::forget(file);
    res.unwrap();
}
