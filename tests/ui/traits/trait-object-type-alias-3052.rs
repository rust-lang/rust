//! Regression test for https://github.com/rust-lang/rust/issues/3052

//@ run-pass
#![allow(dead_code)]

type Connection = Box<dyn FnMut(Vec<u8>) + 'static>;

fn f() -> Option<Connection> {
    let mock_connection: Connection = Box::new(|_| {});
    Some(mock_connection)
}

pub fn main() {}
