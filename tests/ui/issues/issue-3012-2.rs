//@ run-pass
//@ aux-build:issue-3012-1.rs


extern crate socketlib;

use socketlib::socket;

pub fn main() {
    let fd: u32 = 1 as u32;
    let _sock: Box<_> = Box::new(socket::socket_handle(fd));
}
