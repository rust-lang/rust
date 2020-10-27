// run-pass
// ignore-android needs extra network permissions
// ignore-emscripten no threads or sockets support
// ignore-netbsd system ulimit (Too many open files)
// ignore-openbsd system ulimit (Too many open files)

use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};
use std::process;
use std::sync::mpsc::channel;
use std::time::Duration;
use std::thread::{self, Builder};

const TARGET_CNT: usize = 200;

fn main() {
    // This test has a chance to time out, try to not let it time out
    thread::spawn(move|| -> () {
        thread::sleep(Duration::from_secs(30));
        process::exit(1);
    });

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || -> () {
        loop {
            let mut stream = match listener.accept() {
                Ok(stream) => stream.0,
                Err(_) => continue,
            };
            let _ = stream.read(&mut [0]);
            let _ = stream.write(&[2]);
        }
    });

    let (tx, rx) = channel();

    let mut spawned_cnt = 0;
    for _ in 0..TARGET_CNT {
        let tx = tx.clone();
        let res = Builder::new().stack_size(64 * 1024).spawn(move|| {
            match TcpStream::connect(addr) {
                Ok(mut stream) => {
                    let _ = stream.write(&[1]);
                    let _ = stream.read(&mut [0]);
                },
                Err(..) => {}
            }
            tx.send(()).unwrap();
        });
        if let Ok(_) = res {
            spawned_cnt += 1;
        };
    }

    // Wait for all clients to exit, but don't wait for the server to exit. The
    // server just runs infinitely.
    drop(tx);
    for _ in 0..spawned_cnt {
        rx.recv().unwrap();
    }
    assert_eq!(spawned_cnt, TARGET_CNT);
    process::exit(0);
}
