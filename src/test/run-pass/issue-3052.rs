use option::*;

type Connection = fn@(~[u8]);

fn f() -> Option<Connection> {
    let mock_connection: Connection = fn@(_data: ~[u8]) { };
    Some(mock_connection)
}

fn main() {
}
