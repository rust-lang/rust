#![cfg(windows)]
#![feature(windows_unix_domain_sockets)]

use std::io::{Read, Write};
use std::os::windows::net::{UnixListener, UnixStream};
use std::thread;

#[test]
fn smoke_bind_connect() {
    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds.sock");

    let listener = UnixListener::bind(sock_path.as_path()).expect("bind failed");

    let tx = thread::spawn(move || {
        let mut stream = UnixStream::connect(&sock_path).expect("connect failed");
        stream.write_all(b"hello").expect("write failed");
    });

    let (mut stream, _) = listener.accept().expect("accept failed");
    let mut buf = [0; 5];
    stream.read_exact(&mut buf).expect("read failed");
    assert_eq!(&buf, b"hello");

    tx.join().unwrap();

    drop(listener);
    let _ = std::fs::remove_file(&sock_path);
}
