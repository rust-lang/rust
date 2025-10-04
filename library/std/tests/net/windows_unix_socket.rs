#![cfg(all(windows, feature = "windows_unix_domain_sockets"))]
#![unstable(feature = "windows_unix_domain_sockets", issue = "none")]

use std::io::{Read, Write};
use std::os::windows::net::{UnixListener, UnixStream};
use std::path::Path;
use std::thread;

#[test]
fn smoke_bind_connect() {
    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds.sock");

    let listener = UnixListener::bind(&sock_path).expect("bind failed");

    let tx = thread::spawn(move || {
        let mut stream = UnixStream::connect(&sock_path).expect("connect failed");
        stream.write_all(b"hello").expect("write failed");
    });

    let (mut stream, _) = listener.accept().expect("accept failed");
    let mut buf = [0; 5];
    stream.read_exact(&mut buf).expect("read failed");
    assert_eq!(&buf, b"hello");

    tx.join().unwrap;

    drop(listener);
    let _ = std::fs::remove_file(&sock_path);
}

#[test]
fn echo() {
    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds-echo.sock");

    let listener = UnixListener::bind(&sock_path).unwrap();

    let tx = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap;
        let mut buf = [0; 1024];
        loop {
            let n = match stream.read(&mut buf) {
                Ok(0) => return,
                Ok(n) => n,
                Err(e) => panic!("read error: {}", e),
            };
            stream.write_all(&buf[..n]).unwrap;
        }
    });

    let mut client = UnixStream::connect(&sock_path).unwrap;
    client.write_all(b"echo").unwrap;
    let mut buf = [0; 4];
    client.read_exact(&mut buf).unwrap;
    assert_eq!(&buf, b"echo");

    drop(client);
    tx.join().unwrap;
    let _ = std::fs::remove_file(&sock_path);
}

#[test]
fn path_too_long() {
    let long = "\\\\?\\".to_string() + &"a".repeat(300);
    assert!(UnixListener::bind(long).is_err());
}
