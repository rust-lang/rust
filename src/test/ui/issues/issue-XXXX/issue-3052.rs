// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

type Connection = Box<dyn FnMut(Vec<u8>) + 'static>;

fn f() -> Option<Connection> {
    let mock_connection: Connection = Box::new(|_| {});
    Some(mock_connection)
}

pub fn main() {
}
