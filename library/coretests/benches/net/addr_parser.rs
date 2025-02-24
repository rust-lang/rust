use core::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use core::str::FromStr;

use test::{Bencher, black_box};

const IPV4_STR: &str = "192.168.0.1";
const IPV4_STR_PORT: &str = "192.168.0.1:8080";

const IPV6_STR_FULL: &str = "2001:db8:0:0:0:0:c0a8:1";
const IPV6_STR_COMPRESS: &str = "2001:db8::c0a8:1";
const IPV6_STR_V4: &str = "2001:db8::192.168.0.1";
const IPV6_STR_PORT: &str = "[2001:db8::c0a8:1]:8080";
const IPV6_STR_PORT_SCOPE_ID: &str = "[2001:db8::c0a8:1%1337]:8080";

#[bench]
fn bench_parse_ipv4(b: &mut Bencher) {
    b.iter(|| Ipv4Addr::from_str(black_box(IPV4_STR)));
}

#[bench]
fn bench_parse_ipv6_full(b: &mut Bencher) {
    b.iter(|| Ipv6Addr::from_str(black_box(IPV6_STR_FULL)));
}

#[bench]
fn bench_parse_ipv6_compress(b: &mut Bencher) {
    b.iter(|| Ipv6Addr::from_str(black_box(IPV6_STR_COMPRESS)));
}

#[bench]
fn bench_parse_ipv6_v4(b: &mut Bencher) {
    b.iter(|| Ipv6Addr::from_str(black_box(IPV6_STR_V4)));
}

#[bench]
fn bench_parse_ipaddr_v4(b: &mut Bencher) {
    b.iter(|| IpAddr::from_str(black_box(IPV4_STR)));
}

#[bench]
fn bench_parse_ipaddr_v6_full(b: &mut Bencher) {
    b.iter(|| IpAddr::from_str(black_box(IPV6_STR_FULL)));
}

#[bench]
fn bench_parse_ipaddr_v6_compress(b: &mut Bencher) {
    b.iter(|| IpAddr::from_str(black_box(IPV6_STR_COMPRESS)));
}

#[bench]
fn bench_parse_ipaddr_v6_v4(b: &mut Bencher) {
    b.iter(|| IpAddr::from_str(black_box(IPV6_STR_V4)));
}

#[bench]
fn bench_parse_socket_v4(b: &mut Bencher) {
    b.iter(|| SocketAddrV4::from_str(black_box(IPV4_STR_PORT)));
}

#[bench]
fn bench_parse_socket_v6(b: &mut Bencher) {
    b.iter(|| SocketAddrV6::from_str(black_box(IPV6_STR_PORT)));
}

#[bench]
fn bench_parse_socket_v6_scope_id(b: &mut Bencher) {
    b.iter(|| SocketAddrV6::from_str(black_box(IPV6_STR_PORT_SCOPE_ID)));
}

#[bench]
fn bench_parse_socketaddr_v4(b: &mut Bencher) {
    b.iter(|| SocketAddr::from_str(black_box(IPV4_STR_PORT)));
}

#[bench]
fn bench_parse_socketaddr_v6(b: &mut Bencher) {
    b.iter(|| SocketAddr::from_str(black_box(IPV6_STR_PORT)));
}
