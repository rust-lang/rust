//@ run-pass
//@ needs-subprocess

use std::env;
use std::io;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() {
    let mut args = env::args();
    let me = args.next().unwrap();
    let arg = args.next();
    match arg.as_ref().map(|s| &s[..]) {
        None => {
            let mut s = Command::new(&me)
                                .arg("a1")
                                .stdin(Stdio::piped())
                                .spawn()
                                .unwrap();
            s.stdin.take().unwrap().write_all(b"foo\n").unwrap();
            let s = s.wait().unwrap();
            assert!(s.success());
        }
        Some("a1") => {
            let s = Command::new(&me).arg("a2").status().unwrap();
            assert!(s.success());
        }
        Some(..) => {
            let mut s = String::new();
            io::stdin().read_line(&mut s).unwrap();
            assert_eq!(s, "foo\n");
        }
    }
}
