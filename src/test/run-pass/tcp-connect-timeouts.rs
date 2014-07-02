// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty
// compile-flags:--test
// exec-env:RUST_TEST_TASKS=1

// Tests for the connect_timeout() function on a TcpStream. This runs with only
// one test task to ensure that errors are timeouts, not file descriptor
// exhaustion.

#![feature(macro_rules, globs)]
#![allow(experimental)]

extern crate native;
extern crate green;
extern crate rustuv;

#[cfg(test)] #[start]
fn start(argc: int, argv: *const *const u8) -> int {
    green::start(argc, argv, rustuv::event_loop, __test::main)
}

macro_rules! iotest (
    { fn $name:ident() $b:block $(#[$a:meta])* } => (
        mod $name {
            #![allow(unused_imports)]

            use std::io::*;
            use std::io::net::tcp::*;
            use std::io::test::*;
            use std::io;

            fn f() $b

            $(#[$a])* #[test] fn green() { f() }
            $(#[$a])* #[test] fn native() {
                use native;
                let (tx, rx) = channel();
                native::task::spawn(proc() { tx.send(f()) });
                rx.recv();
            }
        }
    )
)

iotest!(fn eventual_timeout() {
    use native;
    let addr = next_test_ip4();
    let host = addr.ip.to_str();
    let port = addr.port;

    // Use a native task to receive connections because it turns out libuv is
    // really good at accepting connections and will likely run out of file
    // descriptors before timing out.
    let (tx1, rx1) = channel();
    let (_tx2, rx2) = channel::<()>();
    native::task::spawn(proc() {
        let _l = TcpListener::bind(host.as_slice(), port).unwrap().listen();
        tx1.send(());
        let _ = rx2.recv_opt();
    });
    rx1.recv();

    let mut v = Vec::new();
    for _ in range(0u, 10000) {
        match TcpStream::connect_timeout(addr, 100) {
            Ok(e) => v.push(e),
            Err(ref e) if e.kind == io::TimedOut => return,
            Err(e) => fail!("other error: {}", e),
        }
    }
    fail!("never timed out!");
} #[ignore(cfg(target_os = "freebsd"))])

iotest!(fn timeout_success() {
    let addr = next_test_ip4();
    let host = addr.ip.to_str();
    let port = addr.port;
    let _l = TcpListener::bind(host.as_slice(), port).unwrap().listen();

    assert!(TcpStream::connect_timeout(addr, 1000).is_ok());
})

iotest!(fn timeout_error() {
    let addr = next_test_ip4();

    assert!(TcpStream::connect_timeout(addr, 1000).is_err());
})
