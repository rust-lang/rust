use std::net::TcpListener;
use std::net::TcpStream;
use std::io::{self, Read, Write};

fn handle_client(stream: TcpStream) -> io::Result<()> {
    stream.write_fmt(format!("message received"))
    //~^ ERROR mismatched types
}

fn main() {
    if let Ok(listener) = TcpListener::bind("127.0.0.1:8080") {
        for incoming in listener.incoming() {
            if let Ok(stream) = incoming {
                handle_client(stream);
            }
        }
    }
}
