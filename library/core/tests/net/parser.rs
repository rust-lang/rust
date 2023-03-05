// FIXME: These tests are all excellent candidates for AFL fuzz testing
use core::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use core::str::FromStr;

const PORT: u16 = 8080;
const SCOPE_ID: u32 = 1337;

const IPV4: Ipv4Addr = Ipv4Addr::new(192, 168, 0, 1);
const IPV4_STR: &str = "192.168.0.1";
const IPV4_STR_PORT: &str = "192.168.0.1:8080";
const IPV4_STR_WITH_OCTAL: &str = "0127.0.0.1";
const IPV4_STR_WITH_HEX: &str = "0x10.0.0.1";

const IPV6: Ipv6Addr = Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0xc0a8, 0x1);
const IPV6_STR_FULL: &str = "2001:db8:0:0:0:0:c0a8:1";
const IPV6_STR_COMPRESS: &str = "2001:db8::c0a8:1";
const IPV6_STR_V4: &str = "2001:db8::192.168.0.1";
const IPV6_STR_V4_WITH_OCTAL: &str = "2001:db8::0127.0.0.1";
const IPV6_STR_V4_WITH_HEX: &str = "2001:db8::0x10.0.0.1";
const IPV6_STR_PORT: &str = "[2001:db8::c0a8:1]:8080";
const IPV6_STR_PORT_SCOPE_ID: &str = "[2001:db8::c0a8:1%1337]:8080";

#[test]
fn parse_ipv4() {
    let result: Ipv4Addr = IPV4_STR.parse().unwrap();
    assert_eq!(result, IPV4);

    assert!(Ipv4Addr::from_str(IPV4_STR_PORT).is_err());
    assert!(Ipv4Addr::from_str(IPV4_STR_WITH_OCTAL).is_err());
    assert!(Ipv4Addr::from_str(IPV4_STR_WITH_HEX).is_err());
    assert!(Ipv4Addr::from_str(IPV6_STR_FULL).is_err());
    assert!(Ipv4Addr::from_str(IPV6_STR_COMPRESS).is_err());
    assert!(Ipv4Addr::from_str(IPV6_STR_V4).is_err());
    assert!(Ipv4Addr::from_str(IPV6_STR_PORT).is_err());
}

#[test]
fn parse_ipv6() {
    let result: Ipv6Addr = IPV6_STR_FULL.parse().unwrap();
    assert_eq!(result, IPV6);

    let result: Ipv6Addr = IPV6_STR_COMPRESS.parse().unwrap();
    assert_eq!(result, IPV6);

    let result: Ipv6Addr = IPV6_STR_V4.parse().unwrap();
    assert_eq!(result, IPV6);

    assert!(Ipv6Addr::from_str(IPV6_STR_V4_WITH_OCTAL).is_err());
    assert!(Ipv6Addr::from_str(IPV6_STR_V4_WITH_HEX).is_err());
    assert!(Ipv6Addr::from_str(IPV4_STR).is_err());
    assert!(Ipv6Addr::from_str(IPV4_STR_PORT).is_err());
    assert!(Ipv6Addr::from_str(IPV6_STR_PORT).is_err());
}

#[test]
fn parse_ip() {
    let result: IpAddr = IPV4_STR.parse().unwrap();
    assert_eq!(result, IpAddr::from(IPV4));

    let result: IpAddr = IPV6_STR_FULL.parse().unwrap();
    assert_eq!(result, IpAddr::from(IPV6));

    let result: IpAddr = IPV6_STR_COMPRESS.parse().unwrap();
    assert_eq!(result, IpAddr::from(IPV6));

    let result: IpAddr = IPV6_STR_V4.parse().unwrap();
    assert_eq!(result, IpAddr::from(IPV6));

    assert!(IpAddr::from_str(IPV4_STR_PORT).is_err());
    assert!(IpAddr::from_str(IPV6_STR_PORT).is_err());
}

#[test]
fn parse_socket_v4() {
    let result: SocketAddrV4 = IPV4_STR_PORT.parse().unwrap();
    assert_eq!(result, SocketAddrV4::new(IPV4, PORT));

    assert!(SocketAddrV4::from_str(IPV4_STR).is_err());
    assert!(SocketAddrV4::from_str(IPV6_STR_FULL).is_err());
    assert!(SocketAddrV4::from_str(IPV6_STR_COMPRESS).is_err());
    assert!(SocketAddrV4::from_str(IPV6_STR_V4).is_err());
    assert!(SocketAddrV4::from_str(IPV6_STR_PORT).is_err());
}

#[test]
fn parse_socket_v6() {
    assert_eq!(IPV6_STR_PORT.parse(), Ok(SocketAddrV6::new(IPV6, PORT, 0, 0)));
    assert_eq!(IPV6_STR_PORT_SCOPE_ID.parse(), Ok(SocketAddrV6::new(IPV6, PORT, 0, SCOPE_ID)));

    assert!(SocketAddrV6::from_str(IPV4_STR).is_err());
    assert!(SocketAddrV6::from_str(IPV4_STR_PORT).is_err());
    assert!(SocketAddrV6::from_str(IPV6_STR_FULL).is_err());
    assert!(SocketAddrV6::from_str(IPV6_STR_COMPRESS).is_err());
    assert!(SocketAddrV6::from_str(IPV6_STR_V4).is_err());
}

#[test]
fn parse_socket() {
    let result: SocketAddr = IPV4_STR_PORT.parse().unwrap();
    assert_eq!(result, SocketAddr::from((IPV4, PORT)));

    let result: SocketAddr = IPV6_STR_PORT.parse().unwrap();
    assert_eq!(result, SocketAddr::from((IPV6, PORT)));

    assert!(SocketAddr::from_str(IPV4_STR).is_err());
    assert!(SocketAddr::from_str(IPV6_STR_FULL).is_err());
    assert!(SocketAddr::from_str(IPV6_STR_COMPRESS).is_err());
    assert!(SocketAddr::from_str(IPV6_STR_V4).is_err());
}

#[test]
fn ipv6_corner_cases() {
    let result: Ipv6Addr = "1::".parse().unwrap();
    assert_eq!(result, Ipv6Addr::new(1, 0, 0, 0, 0, 0, 0, 0));

    let result: Ipv6Addr = "1:1::".parse().unwrap();
    assert_eq!(result, Ipv6Addr::new(1, 1, 0, 0, 0, 0, 0, 0));

    let result: Ipv6Addr = "::1".parse().unwrap();
    assert_eq!(result, Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1));

    let result: Ipv6Addr = "::1:1".parse().unwrap();
    assert_eq!(result, Ipv6Addr::new(0, 0, 0, 0, 0, 0, 1, 1));

    let result: Ipv6Addr = "::".parse().unwrap();
    assert_eq!(result, Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0));

    let result: Ipv6Addr = "::192.168.0.1".parse().unwrap();
    assert_eq!(result, Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0xc0a8, 0x1));

    let result: Ipv6Addr = "::1:192.168.0.1".parse().unwrap();
    assert_eq!(result, Ipv6Addr::new(0, 0, 0, 0, 0, 1, 0xc0a8, 0x1));

    let result: Ipv6Addr = "1:1:1:1:1:1:192.168.0.1".parse().unwrap();
    assert_eq!(result, Ipv6Addr::new(1, 1, 1, 1, 1, 1, 0xc0a8, 0x1));
}

// Things that might not seem like failures but are
#[test]
fn ipv6_corner_failures() {
    // No IP address before the ::
    assert!(Ipv6Addr::from_str("1:192.168.0.1::").is_err());

    // :: must have at least 1 set of zeroes
    assert!(Ipv6Addr::from_str("1:1:1:1::1:1:1:1").is_err());

    // Need brackets for a port
    assert!(SocketAddrV6::from_str("1:1:1:1:1:1:1:1:8080").is_err());
}
