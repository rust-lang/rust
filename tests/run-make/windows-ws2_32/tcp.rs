use std::net::TcpListener;

fn main() {
    TcpListener::bind("127.0.0.1:80").unwrap();
}
