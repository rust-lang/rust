use core::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use core::str::FromStr;

use test::{Bencher, black_box};

const IPV4: &[&str] = &[
    "192.168.0.1",
    "8.8.8.8",
    "127.0.0.1",
    "255.255.255.255",
    "0.0.0.0",
    "10.0.0.1",
    "203.0.113.42",
    "172.16.254.1",
    "100.64.0.1",
    "1.2.3.4",
];

const IPV4_PORT: &[&str] = &[
    "192.168.0.1:8080",
    "8.8.8.8:53",
    "127.0.0.1:65535",
    "255.255.255.255:1",
    "0.0.0.0:80",
    "10.0.0.1:443",
    "203.0.113.42:22",
    "172.16.254.1:3306",
    "100.64.0.1:8443",
    "1.2.3.4:0",
];

const IPV6_FULL: &[&str] = &[
    "2001:db8:0:0:0:0:c0a8:1",
    "2001:db8:85a3:8d3:1319:8a2e:370:7348",
    "fe80:0:0:0:0:0:0:1",
    "ff02:0:0:0:0:0:0:101",
    "2001:4860:4860:0:0:0:0:8888",
    "2606:4700:4700:0:0:0:0:1111",
    "fd00:0:0:0:0:0:0:1",
    "fec0:0:0:0:0:0:0:abcd",
    "1:2:3:4:5:6:7:8",
    "abcd:ef01:2345:6789:abcd:ef01:2345:6789",
];

const IPV6_COMPRESS: &[&str] = &[
    "2001:db8::c0a8:1",
    "::1",
    "fe80::1",
    "2001:db8::",
    "ff02::1:2",
    "64:ff9b::",
    "::",
    "2001:4860:4860::8888",
    "2606:4700:4700::1111",
    "fe80::1ff:fe23:4567:890a",
];

const IPV6_V4: &[&str] = &[
    "2001:db8::192.168.0.1",
    "::ffff:192.168.0.1",
    "64:ff9b::192.0.2.33",
    "::ffff:8.8.8.8",
    "::192.168.0.1",
    "::ffff:255.255.255.255",
    "2001:db8:0:0:0:0:192.168.0.1",
    "64:ff9b::10.0.0.1",
    "::ffff:1.2.3.4",
    "::ffff:127.0.0.1",
];

const IPV6_PORT: &[&str] = &[
    "[2001:db8::c0a8:1]:8080",
    "[::1]:443",
    "[fe80::1]:53",
    "[2001:db8:85a3:8d3:1319:8a2e:370:7348]:22",
    "[::]:80",
    "[2001:4860:4860::8888]:443",
    "[2606:4700:4700::1111]:53",
    "[fe80::1ff:fe23:4567:890a]:8080",
    "[64:ff9b::192.0.2.33]:443",
    "[::ffff:8.8.8.8]:53",
];

const IPV6_PORT_SCOPE_ID: &[&str] = &[
    "[2001:db8::c0a8:1%1337]:8080",
    "[fe80::1%1]:53",
    "[fe80::1%999999]:443",
    "[fe80::1%0]:80",
    "[fe80::1ff:fe23:4567:890a%2]:8080",
    "[::1%1]:443",
    "[fe80::abcd%15]:22",
    "[ff02::1%42]:5353",
    "[fe80::1%4294967295]:443",
    "[2001:db8::1%100]:8080",
];

#[bench]
fn bench_parse_ipv4(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV4 {
            let _ = black_box(Ipv4Addr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_ipv6_full(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_FULL {
            let _ = black_box(Ipv6Addr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_ipv6_compress(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_COMPRESS {
            let _ = black_box(Ipv6Addr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_ipv6_v4(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_V4 {
            let _ = black_box(Ipv6Addr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_ipaddr_v4(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV4 {
            let _ = black_box(IpAddr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_ipaddr_v6_full(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_FULL {
            let _ = black_box(IpAddr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_ipaddr_v6_compress(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_COMPRESS {
            let _ = black_box(IpAddr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_ipaddr_v6_v4(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_V4 {
            let _ = black_box(IpAddr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_socket_v4(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV4_PORT {
            let _ = black_box(SocketAddrV4::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_socket_v6(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_PORT {
            let _ = black_box(SocketAddrV6::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_socket_v6_scope_id(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_PORT_SCOPE_ID {
            let _ = black_box(SocketAddrV6::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_socketaddr_v4(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV4_PORT {
            let _ = black_box(SocketAddr::from_str(black_box(s)));
        }
    });
}

#[bench]
fn bench_parse_socketaddr_v6(b: &mut Bencher) {
    b.iter(|| {
        for s in IPV6_PORT {
            let _ = black_box(SocketAddr::from_str(black_box(s)));
        }
    });
}
