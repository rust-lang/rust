#![cfg(windows)]
#![feature(windows_unix_domain_sockets)]

use std::io::{Read, Write};
use std::os::windows::net::{UnixListener, UnixStream};
use std::thread;

#[test]
fn smoke_bind_connect() {
    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds.sock");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).expect("bind failed");
    let sock_path_clone = sock_path.clone();
    let tx = thread::spawn(move || {
        let mut stream = UnixStream::connect(&sock_path_clone).expect("connect failed");
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

#[test]
fn win_uds_echo() {
    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds-echo.sock");
    let _ = std::fs::remove_file(&sock_path);

    let listener = UnixListener::bind(&sock_path).expect("bind failed");
    let srv = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept failed");
        let mut buf = [0u8; 128];
        loop {
            let n = match stream.read(&mut buf) {
                Ok(0) => break, // 对端关闭
                Ok(n) => n,
                Err(e) => panic!("read error: {}", e),
            };
            stream.write_all(&buf[..n]).expect("write_all failed");
        }
    });

    let sock_path_clone = sock_path.clone();
    let cli = thread::spawn(move || {
        let mut stream = UnixStream::connect(&sock_path_clone).expect("connect failed");
        let req = b"hello windows uds";
        stream.write_all(req).expect("write failed");
        let mut resp = vec![0u8; req.len()];
        stream.read_exact(&mut resp).expect("read failed");
        assert_eq!(resp, req);
    });

    cli.join().unwrap();
    srv.join().unwrap();

    let _ = std::fs::remove_file(&sock_path);
}
