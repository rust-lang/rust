//@ run-pass

use std::{fs, net, sync};

fn assert_both<T: Sync + Send>() {}

fn main() {
    assert_both::<sync::Mutex<()>>();
    assert_both::<sync::Condvar>();
    assert_both::<sync::RwLock<()>>();
    assert_both::<sync::Barrier>();
    assert_both::<sync::Arc<()>>();
    assert_both::<sync::Weak<()>>();
    assert_both::<sync::Once>();

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
