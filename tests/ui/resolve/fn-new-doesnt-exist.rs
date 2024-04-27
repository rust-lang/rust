use std::net::TcpStream;

fn main() {
   let stream = TcpStream::new(); //~ ERROR no function or associated item named `new` found
}
