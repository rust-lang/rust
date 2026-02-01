#![cfg(windows)]
#![feature(windows_unix_domain_sockets)]
// Now only test windows_unix_domain_sockets feature
// in the future, will test both unix and windows uds
use std::io::{Read, Write};
use std::os::windows::net::{UnixListener, UnixStream};
use std::thread;

#[test]
fn win_uds_smoke_bind_connect() {
    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds-smoke.sock");
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
                Ok(0) => break,
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

#[test]
fn win_uds_path_too_long() {
    let tmp = std::env::temp_dir();
    let long_path = tmp.join("a".repeat(200));
    let result = UnixListener::bind(&long_path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(&long_path);
}
#[test]
fn win_uds_existing_bind() {
    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds-existing.sock");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).expect("bind failed");
    let result = UnixListener::bind(&sock_path);
    assert!(result.is_err());
    drop(listener);
    let _ = std::fs::remove_file(&sock_path);
}
