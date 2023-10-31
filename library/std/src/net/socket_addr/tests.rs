use crate::net::test::{sa4, sa6, tsa};
use crate::net::*;

#[test]
fn to_socket_addr_ipaddr_u16() {
    let a = Ipv4Addr::new(77, 88, 21, 11);
    let p = 12345;
    let e = SocketAddr::V4(SocketAddrV4::new(a, p));
    assert_eq!(Ok(vec![e]), tsa((a, p)));
}

#[test]
fn to_socket_addr_str_u16() {
    let a = sa4(Ipv4Addr::new(77, 88, 21, 11), 24352);
    assert_eq!(Ok(vec![a]), tsa(("77.88.21.11", 24352)));

    let a = sa6(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53);
    assert_eq!(Ok(vec![a]), tsa(("2a02:6b8:0:1::1", 53)));

    let a = sa4(Ipv4Addr::new(127, 0, 0, 1), 23924);
    #[cfg(not(target_env = "sgx"))]
    assert!(tsa(("localhost", 23924)).unwrap().contains(&a));
    #[cfg(target_env = "sgx")]
    let _ = a;
}

#[test]
fn to_socket_addr_str() {
    let a = sa4(Ipv4Addr::new(77, 88, 21, 11), 24352);
    assert_eq!(Ok(vec![a]), tsa("77.88.21.11:24352"));

    let a = sa6(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53);
    assert_eq!(Ok(vec![a]), tsa("[2a02:6b8:0:1::1]:53"));

    let a = sa4(Ipv4Addr::new(127, 0, 0, 1), 23924);
    #[cfg(not(target_env = "sgx"))]
    assert!(tsa("localhost:23924").unwrap().contains(&a));
    #[cfg(target_env = "sgx")]
    let _ = a;
}

#[test]
fn to_socket_addr_string() {
    let a = sa4(Ipv4Addr::new(77, 88, 21, 11), 24352);
    assert_eq!(Ok(vec![a]), tsa(&*format!("{}:{}", "77.88.21.11", "24352")));
    assert_eq!(Ok(vec![a]), tsa(&format!("{}:{}", "77.88.21.11", "24352")));
    assert_eq!(Ok(vec![a]), tsa(format!("{}:{}", "77.88.21.11", "24352")));

    let s = format!("{}:{}", "77.88.21.11", "24352");
    assert_eq!(Ok(vec![a]), tsa(s));
    // s has been moved into the tsa call
}

#[test]
fn ipv4_socket_addr_to_string() {
    // Shortest possible IPv4 length.
    assert_eq!(SocketAddrV4::new(Ipv4Addr::new(0, 0, 0, 0), 0).to_string(), "0.0.0.0:0");

    // Longest possible IPv4 length.
    assert_eq!(
        SocketAddrV4::new(Ipv4Addr::new(255, 255, 255, 255), u16::MAX).to_string(),
        "255.255.255.255:65535"
    );

    // Test padding.
    assert_eq!(
        format!("{:16}", SocketAddrV4::new(Ipv4Addr::new(1, 1, 1, 1), 53)),
        "1.1.1.1:53      "
    );
    assert_eq!(
        format!("{:>16}", SocketAddrV4::new(Ipv4Addr::new(1, 1, 1, 1), 53)),
        "      1.1.1.1:53"
    );
}

#[test]
fn ipv6_socket_addr_to_string() {
    // IPv4-mapped address.
    assert_eq!(
        SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc000, 0x280), 8080, 0, 0)
            .to_string(),
        "[::ffff:192.0.2.128]:8080"
    );

    // IPv4-compatible address.
    assert_eq!(
        SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0xc000, 0x280), 8080, 0, 0).to_string(),
        "[::c000:280]:8080"
    );

    // IPv6 address with no zero segments.
    assert_eq!(
        SocketAddrV6::new(Ipv6Addr::new(8, 9, 10, 11, 12, 13, 14, 15), 80, 0, 0).to_string(),
        "[8:9:a:b:c:d:e:f]:80"
    );

    // Shortest possible IPv6 length.
    assert_eq!(SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, 0, 0, 0).to_string(), "[::]:0");

    // Longest possible IPv6 length.
    assert_eq!(
        SocketAddrV6::new(
            Ipv6Addr::new(0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888),
            u16::MAX,
            u32::MAX,
            u32::MAX,
        )
        .to_string(),
        "[1111:2222:3333:4444:5555:6666:7777:8888%4294967295]:65535"
    );

    // Test padding.
    assert_eq!(
        format!("{:22}", SocketAddrV6::new(Ipv6Addr::new(1, 2, 3, 4, 5, 6, 7, 8), 9, 0, 0)),
        "[1:2:3:4:5:6:7:8]:9   "
    );
    assert_eq!(
        format!("{:>22}", SocketAddrV6::new(Ipv6Addr::new(1, 2, 3, 4, 5, 6, 7, 8), 9, 0, 0)),
        "   [1:2:3:4:5:6:7:8]:9"
    );
}

#[test]
fn bind_udp_socket_bad() {
    // rust-lang/rust#53957: This is a regression test for a parsing problem
    // discovered as part of issue rust-lang/rust#23076, where we were
    // incorrectly parsing invalid input and then that would result in a
    // successful `UdpSocket` binding when we would expect failure.
    //
    // At one time, this test was written as a call to `tsa` with
    // INPUT_23076. However, that structure yields an unreliable test,
    // because it ends up passing junk input to the DNS server, and some DNS
    // servers will respond with `Ok` to such input, with the ip address of
    // the DNS server itself.
    //
    // This form of the test is more robust: even when the DNS server
    // returns its own address, it is still an error to bind a UDP socket to
    // a non-local address, and so we still get an error here in that case.

    const INPUT_23076: &str = "1200::AB00:1234::2552:7777:1313:34300";

    assert!(crate::net::UdpSocket::bind(INPUT_23076).is_err())
}

#[test]
fn set_ip() {
    fn ip4(low: u8) -> Ipv4Addr {
        Ipv4Addr::new(77, 88, 21, low)
    }
    fn ip6(low: u16) -> Ipv6Addr {
        Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, low)
    }

    let mut v4 = SocketAddrV4::new(ip4(11), 80);
    assert_eq!(v4.ip(), &ip4(11));
    v4.set_ip(ip4(12));
    assert_eq!(v4.ip(), &ip4(12));

    let mut addr = SocketAddr::V4(v4);
    assert_eq!(addr.ip(), IpAddr::V4(ip4(12)));
    addr.set_ip(IpAddr::V4(ip4(13)));
    assert_eq!(addr.ip(), IpAddr::V4(ip4(13)));
    addr.set_ip(IpAddr::V6(ip6(14)));
    assert_eq!(addr.ip(), IpAddr::V6(ip6(14)));

    let mut v6 = SocketAddrV6::new(ip6(1), 80, 0, 0);
    assert_eq!(v6.ip(), &ip6(1));
    v6.set_ip(ip6(2));
    assert_eq!(v6.ip(), &ip6(2));

    let mut addr = SocketAddr::V6(v6);
    assert_eq!(addr.ip(), IpAddr::V6(ip6(2)));
    addr.set_ip(IpAddr::V6(ip6(3)));
    assert_eq!(addr.ip(), IpAddr::V6(ip6(3)));
    addr.set_ip(IpAddr::V4(ip4(4)));
    assert_eq!(addr.ip(), IpAddr::V4(ip4(4)));
}

#[test]
fn set_port() {
    let mut v4 = SocketAddrV4::new(Ipv4Addr::new(77, 88, 21, 11), 80);
    assert_eq!(v4.port(), 80);
    v4.set_port(443);
    assert_eq!(v4.port(), 443);

    let mut addr = SocketAddr::V4(v4);
    assert_eq!(addr.port(), 443);
    addr.set_port(8080);
    assert_eq!(addr.port(), 8080);

    let mut v6 = SocketAddrV6::new(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 80, 0, 0);
    assert_eq!(v6.port(), 80);
    v6.set_port(443);
    assert_eq!(v6.port(), 443);

    let mut addr = SocketAddr::V6(v6);
    assert_eq!(addr.port(), 443);
    addr.set_port(8080);
    assert_eq!(addr.port(), 8080);
}

#[test]
fn set_flowinfo() {
    let mut v6 = SocketAddrV6::new(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 80, 10, 0);
    assert_eq!(v6.flowinfo(), 10);
    v6.set_flowinfo(20);
    assert_eq!(v6.flowinfo(), 20);
}

#[test]
fn set_scope_id() {
    let mut v6 = SocketAddrV6::new(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 80, 0, 10);
    assert_eq!(v6.scope_id(), 10);
    v6.set_scope_id(20);
    assert_eq!(v6.scope_id(), 20);
}

#[test]
fn is_v4() {
    let v4 = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(77, 88, 21, 11), 80));
    assert!(v4.is_ipv4());
    assert!(!v4.is_ipv6());
}

#[test]
fn is_v6() {
    let v6 = SocketAddr::V6(SocketAddrV6::new(
        Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1),
        80,
        10,
        0,
    ));
    assert!(!v6.is_ipv4());
    assert!(v6.is_ipv6());
}

#[test]
fn socket_v4_to_str() {
    let socket = SocketAddrV4::new(Ipv4Addr::new(192, 168, 0, 1), 8080);

    assert_eq!(format!("{socket}"), "192.168.0.1:8080");
    assert_eq!(format!("{socket:<20}"), "192.168.0.1:8080    ");
    assert_eq!(format!("{socket:>20}"), "    192.168.0.1:8080");
    assert_eq!(format!("{socket:^20}"), "  192.168.0.1:8080  ");
    assert_eq!(format!("{socket:.10}"), "192.168.0.");
}

#[test]
fn socket_v6_to_str() {
    let mut socket = SocketAddrV6::new(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53, 0, 0);

    assert_eq!(format!("{socket}"), "[2a02:6b8:0:1::1]:53");
    assert_eq!(format!("{socket:<24}"), "[2a02:6b8:0:1::1]:53    ");
    assert_eq!(format!("{socket:>24}"), "    [2a02:6b8:0:1::1]:53");
    assert_eq!(format!("{socket:^24}"), "  [2a02:6b8:0:1::1]:53  ");
    assert_eq!(format!("{socket:.15}"), "[2a02:6b8:0:1::");

    socket.set_scope_id(5);

    assert_eq!(format!("{socket}"), "[2a02:6b8:0:1::1%5]:53");
    assert_eq!(format!("{socket:<24}"), "[2a02:6b8:0:1::1%5]:53  ");
    assert_eq!(format!("{socket:>24}"), "  [2a02:6b8:0:1::1%5]:53");
    assert_eq!(format!("{socket:^24}"), " [2a02:6b8:0:1::1%5]:53 ");
    assert_eq!(format!("{socket:.18}"), "[2a02:6b8:0:1::1%5");
}

#[test]
fn compare() {
    let v4_1 = "224.120.45.1:23456".parse::<SocketAddrV4>().unwrap();
    let v4_2 = "224.210.103.5:12345".parse::<SocketAddrV4>().unwrap();
    let v4_3 = "224.210.103.5:23456".parse::<SocketAddrV4>().unwrap();
    let v6_1 = "[2001:db8:f00::1002]:23456".parse::<SocketAddrV6>().unwrap();
    let v6_2 = "[2001:db8:f00::2001]:12345".parse::<SocketAddrV6>().unwrap();
    let v6_3 = "[2001:db8:f00::2001]:23456".parse::<SocketAddrV6>().unwrap();

    // equality
    assert_eq!(v4_1, v4_1);
    assert_eq!(v6_1, v6_1);
    assert_eq!(SocketAddr::V4(v4_1), SocketAddr::V4(v4_1));
    assert_eq!(SocketAddr::V6(v6_1), SocketAddr::V6(v6_1));
    assert!(v4_1 != v4_2);
    assert!(v6_1 != v6_2);

    // compare different addresses
    assert!(v4_1 < v4_2);
    assert!(v6_1 < v6_2);
    assert!(v4_2 > v4_1);
    assert!(v6_2 > v6_1);

    // compare the same address with different ports
    assert!(v4_2 < v4_3);
    assert!(v6_2 < v6_3);
    assert!(v4_3 > v4_2);
    assert!(v6_3 > v6_2);

    // compare different addresses with the same port
    assert!(v4_1 < v4_3);
    assert!(v6_1 < v6_3);
    assert!(v4_3 > v4_1);
    assert!(v6_3 > v6_1);

    // compare with an inferred right-hand side
    assert_eq!(v4_1, "224.120.45.1:23456".parse().unwrap());
    assert_eq!(v6_1, "[2001:db8:f00::1002]:23456".parse().unwrap());
    assert_eq!(SocketAddr::V4(v4_1), "224.120.45.1:23456".parse().unwrap());
}
