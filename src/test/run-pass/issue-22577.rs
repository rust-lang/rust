// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{fs, net};

fn assert_both<T: Send + Sync>() {}

fn main() {
    assert_both::<fs::File>();
    assert_both::<fs::Metadata>();
    assert_both::<fs::ReadDir>();
    assert_both::<fs::DirEntry>();
    assert_both::<fs::WalkDir>();
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
