//@ run-rustfix
use std::net::TcpListener;

struct NoToSocketAddrs(String);

impl std::ops::Deref for NoToSocketAddrs {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn main() {
    let _works = TcpListener::bind("some string");
    let bad = NoToSocketAddrs("bad".to_owned());
    let _errors = TcpListener::bind(&bad);
    //~^ ERROR the trait bound `NoToSocketAddrs: ToSocketAddrs` is not satisfied
}
