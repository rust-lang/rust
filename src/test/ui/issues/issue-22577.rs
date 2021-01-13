// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

use std::{fs, net};

fn assert_both<T: Send + Sync>() {}
fn assert_send<T: Send>() {}

fn main() {
    assert_both::<fs::File>();
    assert_both::<fs::Metadata>();
    assert_both::<fs::ReadDir>();
    assert_both::<fs::DirEntry>();
    assert_both::<fs::OpenOptions>();
    assert_both::<fs::Permissions>();

    assert_both::<net::TcpStream>();
    assert_both::<net::TcpListener>();
    assert_both::<net::UdpSocket>();
    assert_both::<net::SocketAddr>();
    assert_both::<net::SocketAddrV4>();
    assert_both::<net::SocketAddrV6>();
    assert_both::<net::Ipv4Addr>();
    assert_both::<net::Ipv6Addr>();
}
