//@ run-pass
//@ needs-threads
//@ needs-subprocess

use std::process::Command;
use std::{env, thread};

struct Handle(i32);

impl Drop for Handle {
    fn drop(&mut self) {
        panic!();
    }
}

thread_local!(static HANDLE: Handle = Handle(0));

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 1 {
        let out = Command::new(&args[0]).arg("test").output().unwrap();
        let stderr = std::str::from_utf8(&out.stderr).unwrap();
        assert!(stderr.contains("explicit panic"), "bad failure message:\n{}\n", stderr);
    } else {
        // TLS dtors are not always run on process exit
        thread::spawn(|| {
            HANDLE.with(|h| {
                println!("{}", h.0);
            });
        })
        .join()
        .unwrap();
    }
}
