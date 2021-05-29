use crate::net::test::{sa4, sa6, tsa};
use crate::net::*;
use crate::str::FromStr;

#[test]
fn test_from_str_ipv4() {
    assert_eq!(Ok(Ipv4Addr::new(127, 0, 0, 1)), "127.0.0.1".parse());
    assert_eq!(Ok(Ipv4Addr::new(255, 255, 255, 255)), "255.255.255.255".parse());
    assert_eq!(Ok(Ipv4Addr::new(0, 0, 0, 0)), "0.0.0.0".parse());

    // out of range
    let none: Option<Ipv4Addr> = "256.0.0.1".parse().ok();
    assert_eq!(None, none);
    // too short
    let none: Option<Ipv4Addr> = "255.0.0".parse().ok();
    assert_eq!(None, none);
    // too long
    let none: Option<Ipv4Addr> = "255.0.0.1.2".parse().ok();
    assert_eq!(None, none);
    // no number between dots
    let none: Option<Ipv4Addr> = "255.0..1".parse().ok();
    assert_eq!(None, none);
}

#[test]
fn test_from_str_ipv6() {
    assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0)), "0:0:0:0:0:0:0:0".parse());
    assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)), "0:0:0:0:0:0:0:1".parse());

    assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)), "::1".parse());
    assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0)), "::".parse());

    assert_eq!(Ok(Ipv6Addr::new(0x2a02, 0x6b8, 0, 0, 0, 0, 0x11, 0x11)), "2a02:6b8::11:11".parse());

    // too long group
    let none: Option<Ipv6Addr> = "::00000".parse().ok();
    assert_eq!(None, none);
    // too short
    let none: Option<Ipv6Addr> = "1:2:3:4:5:6:7".parse().ok();
    assert_eq!(None, none);
    // too long
    let none: Option<Ipv6Addr> = "1:2:3:4:5:6:7:8:9".parse().ok();
    assert_eq!(None, none);
    // triple colon
    let none: Option<Ipv6Addr> = "1:2:::6:7:8".parse().ok();
    assert_eq!(None, none);
    // two double colons
    let none: Option<Ipv6Addr> = "1:2::6::8".parse().ok();
    assert_eq!(None, none);
    // `::` indicating zero groups of zeros
    let none: Option<Ipv6Addr> = "1:2:3:4::5:6:7:8".parse().ok();
    assert_eq!(None, none);
}

#[test]
fn test_from_str_ipv4_in_ipv6() {
    assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 49152, 545)), "::192.0.2.33".parse());
    assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0xFFFF, 49152, 545)), "::FFFF:192.0.2.33".parse());
    assert_eq!(
        Ok(Ipv6Addr::new(0x64, 0xff9b, 0, 0, 0, 0, 49152, 545)),
        "64:ff9b::192.0.2.33".parse()
    );
    assert_eq!(
        Ok(Ipv6Addr::new(0x2001, 0xdb8, 0x122, 0xc000, 0x2, 0x2100, 49152, 545)),
        "2001:db8:122:c000:2:2100:192.0.2.33".parse()
    );

    // colon after v4
    let none: Option<Ipv4Addr> = "::127.0.0.1:".parse().ok();
    assert_eq!(None, none);
    // not enough groups
    let none: Option<Ipv6Addr> = "1.2.3.4.5:127.0.0.1".parse().ok();
    assert_eq!(None, none);
    // too many groups
    let none: Option<Ipv6Addr> = "1.2.3.4.5:6:7:127.0.0.1".parse().ok();
    assert_eq!(None, none);
}

#[test]
fn test_from_str_socket_addr() {
    assert_eq!(Ok(sa4(Ipv4Addr::new(77, 88, 21, 11), 80)), "77.88.21.11:80".parse());
    assert_eq!(Ok(SocketAddrV4::new(Ipv4Addr::new(77, 88, 21, 11), 80)), "77.88.21.11:80".parse());
    assert_eq!(
        Ok(sa6(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53)),
        "[2a02:6b8:0:1::1]:53".parse()
    );
    assert_eq!(
        Ok(SocketAddrV6::new(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53, 0, 0)),
        "[2a02:6b8:0:1::1]:53".parse()
    );
    assert_eq!(Ok(sa6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x7F00, 1), 22)), "[::127.0.0.1]:22".parse());
    assert_eq!(
        Ok(SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x7F00, 1), 22, 0, 0)),
        "[::127.0.0.1]:22".parse()
    );

    // without port
    let none: Option<SocketAddr> = "127.0.0.1".parse().ok();
    assert_eq!(None, none);
    // without port
    let none: Option<SocketAddr> = "127.0.0.1:".parse().ok();
    assert_eq!(None, none);
    // wrong brackets around v4
    let none: Option<SocketAddr> = "[127.0.0.1]:22".parse().ok();
    assert_eq!(None, none);
    // port out of range
    let none: Option<SocketAddr> = "127.0.0.1:123456".parse().ok();
    assert_eq!(None, none);
}

#[test]
fn ipv4_addr_to_string() {
    assert_eq!(Ipv4Addr::new(127, 0, 0, 1).to_string(), "127.0.0.1");
    // Short address
    assert_eq!(Ipv4Addr::new(1, 1, 1, 1).to_string(), "1.1.1.1");
    // Long address
    assert_eq!(Ipv4Addr::new(127, 127, 127, 127).to_string(), "127.127.127.127");

    // Test padding
    assert_eq!(&format!("{:16}", Ipv4Addr::new(1, 1, 1, 1)), "1.1.1.1         ");
    assert_eq!(&format!("{:>16}", Ipv4Addr::new(1, 1, 1, 1)), "         1.1.1.1");
}

#[test]
fn ipv6_addr_to_string() {
    // ipv4-mapped address
    let a1 = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc000, 0x280);
    assert_eq!(a1.to_string(), "::ffff:192.0.2.128");

    // ipv4-compatible address
    let a1 = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0xc000, 0x280);
    assert_eq!(a1.to_string(), "::192.0.2.128");

    // v6 address with no zero segments
    assert_eq!(Ipv6Addr::new(8, 9, 10, 11, 12, 13, 14, 15).to_string(), "8:9:a:b:c:d:e:f");

    // longest possible IPv6 length
    assert_eq!(
        Ipv6Addr::new(0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888).to_string(),
        "1111:2222:3333:4444:5555:6666:7777:8888"
    );
    // padding
    assert_eq!(&format!("{:20}", Ipv6Addr::new(1, 2, 3, 4, 5, 6, 7, 8)), "1:2:3:4:5:6:7:8     ");
    assert_eq!(&format!("{:>20}", Ipv6Addr::new(1, 2, 3, 4, 5, 6, 7, 8)), "     1:2:3:4:5:6:7:8");

    // reduce a single run of zeros
    assert_eq!(
        "ae::ffff:102:304",
        Ipv6Addr::new(0xae, 0, 0, 0, 0, 0xffff, 0x0102, 0x0304).to_string()
    );

    // don't reduce just a single zero segment
    assert_eq!("1:2:3:4:5:6:0:8", Ipv6Addr::new(1, 2, 3, 4, 5, 6, 0, 8).to_string());

    // 'any' address
    assert_eq!("::", Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0).to_string());

    // loopback address
    assert_eq!("::1", Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1).to_string());

    // ends in zeros
    assert_eq!("1::", Ipv6Addr::new(1, 0, 0, 0, 0, 0, 0, 0).to_string());

    // two runs of zeros, second one is longer
    assert_eq!("1:0:0:4::8", Ipv6Addr::new(1, 0, 0, 4, 0, 0, 0, 8).to_string());

    // two runs of zeros, equal length
    assert_eq!("1::4:5:0:0:8", Ipv6Addr::new(1, 0, 0, 4, 5, 0, 0, 8).to_string());

    // don't prefix `0x` to each segment in `dbg!`.
    assert_eq!("1::4:5:0:0:8", &format!("{:#?}", Ipv6Addr::new(1, 0, 0, 4, 5, 0, 0, 8)));
}

#[test]
fn ipv4_to_ipv6() {
    assert_eq!(
        Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x1234, 0x5678),
        Ipv4Addr::new(0x12, 0x34, 0x56, 0x78).to_ipv6_mapped()
    );
    assert_eq!(
        Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x1234, 0x5678),
        Ipv4Addr::new(0x12, 0x34, 0x56, 0x78).to_ipv6_compatible()
    );
}

#[test]
fn ipv6_to_ipv4_mapped() {
    assert_eq!(
        Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x1234, 0x5678).to_ipv4_mapped(),
        Some(Ipv4Addr::new(0x12, 0x34, 0x56, 0x78))
    );
    assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x1234, 0x5678).to_ipv4_mapped(), None);
}

#[test]
fn ipv6_to_ipv4() {
    assert_eq!(
        Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x1234, 0x5678).to_ipv4(),
        Some(Ipv4Addr::new(0x12, 0x34, 0x56, 0x78))
    );
    assert_eq!(
        Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x1234, 0x5678).to_ipv4(),
        Some(Ipv4Addr::new(0x12, 0x34, 0x56, 0x78))
    );
    assert_eq!(Ipv6Addr::new(0, 0, 1, 0, 0, 0, 0x1234, 0x5678).to_ipv4(), None);
}

#[test]
fn ip_properties() {
    macro_rules! ip {
        ($s:expr) => {
            IpAddr::from_str($s).unwrap()
        };
    }

    macro_rules! check {
        ($s:expr) => {
            check!($s, 0);
        };

        ($s:expr, $mask:expr) => {{
            let unspec: u8 = 1 << 0;
            let loopback: u8 = 1 << 1;
            let global: u8 = 1 << 2;
            let multicast: u8 = 1 << 3;
            let doc: u8 = 1 << 4;

            if ($mask & unspec) == unspec {
                assert!(ip!($s).is_unspecified());
            } else {
                assert!(!ip!($s).is_unspecified());
            }

            if ($mask & loopback) == loopback {
                assert!(ip!($s).is_loopback());
            } else {
                assert!(!ip!($s).is_loopback());
            }

            if ($mask & global) == global {
                assert!(ip!($s).is_global());
            } else {
                assert!(!ip!($s).is_global());
            }

            if ($mask & multicast) == multicast {
                assert!(ip!($s).is_multicast());
            } else {
                assert!(!ip!($s).is_multicast());
            }

            if ($mask & doc) == doc {
                assert!(ip!($s).is_documentation());
            } else {
                assert!(!ip!($s).is_documentation());
            }
        }};
    }

    let unspec: u8 = 1 << 0;
    let loopback: u8 = 1 << 1;
    let global: u8 = 1 << 2;
    let multicast: u8 = 1 << 3;
    let doc: u8 = 1 << 4;

    check!("0.0.0.0", unspec);
    check!("0.0.0.1");
    check!("0.1.0.0");
    check!("10.9.8.7");
    check!("127.1.2.3", loopback);
    check!("172.31.254.253");
    check!("169.254.253.242");
    check!("192.0.2.183", doc);
    check!("192.1.2.183", global);
    check!("192.168.254.253");
    check!("198.51.100.0", doc);
    check!("203.0.113.0", doc);
    check!("203.2.113.0", global);
    check!("224.0.0.0", global | multicast);
    check!("239.255.255.255", global | multicast);
    check!("255.255.255.255");
    // make sure benchmarking addresses are not global
    check!("198.18.0.0");
    check!("198.18.54.2");
    check!("198.19.255.255");
    // make sure addresses reserved for protocol assignment are not global
    check!("192.0.0.0");
    check!("192.0.0.255");
    check!("192.0.0.100");
    // make sure reserved addresses are not global
    check!("240.0.0.0");
    check!("251.54.1.76");
    check!("254.255.255.255");
    // make sure shared addresses are not global
    check!("100.64.0.0");
    check!("100.127.255.255");
    check!("100.100.100.0");

    check!("::", unspec);
    check!("::1", loopback);
    check!("::0.0.0.2", global);
    check!("1::", global);
    check!("fc00::");
    check!("fdff:ffff::");
    check!("fe80:ffff::");
    check!("febf:ffff::");
    check!("fec0::", global);
    check!("ff01::", multicast);
    check!("ff02::", multicast);
    check!("ff03::", multicast);
    check!("ff04::", multicast);
    check!("ff05::", multicast);
    check!("ff08::", multicast);
    check!("ff0e::", global | multicast);
    check!("2001:db8:85a3::8a2e:370:7334", doc);
    check!("102:304:506:708:90a:b0c:d0e:f10", global);
}

#[test]
fn ipv4_properties() {
    macro_rules! ip {
        ($s:expr) => {
            Ipv4Addr::from_str($s).unwrap()
        };
    }

    macro_rules! check {
        ($s:expr) => {
            check!($s, 0);
        };

        ($s:expr, $mask:expr) => {{
            let unspec: u16 = 1 << 0;
            let loopback: u16 = 1 << 1;
            let private: u16 = 1 << 2;
            let link_local: u16 = 1 << 3;
            let global: u16 = 1 << 4;
            let multicast: u16 = 1 << 5;
            let broadcast: u16 = 1 << 6;
            let documentation: u16 = 1 << 7;
            let benchmarking: u16 = 1 << 8;
            let ietf_protocol_assignment: u16 = 1 << 9;
            let reserved: u16 = 1 << 10;
            let shared: u16 = 1 << 11;

            if ($mask & unspec) == unspec {
                assert!(ip!($s).is_unspecified());
            } else {
                assert!(!ip!($s).is_unspecified());
            }

            if ($mask & loopback) == loopback {
                assert!(ip!($s).is_loopback());
            } else {
                assert!(!ip!($s).is_loopback());
            }

            if ($mask & private) == private {
                assert!(ip!($s).is_private());
            } else {
                assert!(!ip!($s).is_private());
            }

            if ($mask & link_local) == link_local {
                assert!(ip!($s).is_link_local());
            } else {
                assert!(!ip!($s).is_link_local());
            }

            if ($mask & global) == global {
                assert!(ip!($s).is_global());
            } else {
                assert!(!ip!($s).is_global());
            }

            if ($mask & multicast) == multicast {
                assert!(ip!($s).is_multicast());
            } else {
                assert!(!ip!($s).is_multicast());
            }

            if ($mask & broadcast) == broadcast {
                assert!(ip!($s).is_broadcast());
            } else {
                assert!(!ip!($s).is_broadcast());
            }

            if ($mask & documentation) == documentation {
                assert!(ip!($s).is_documentation());
            } else {
                assert!(!ip!($s).is_documentation());
            }

            if ($mask & benchmarking) == benchmarking {
                assert!(ip!($s).is_benchmarking());
            } else {
                assert!(!ip!($s).is_benchmarking());
            }

            if ($mask & ietf_protocol_assignment) == ietf_protocol_assignment {
                assert!(ip!($s).is_ietf_protocol_assignment());
            } else {
                assert!(!ip!($s).is_ietf_protocol_assignment());
            }

            if ($mask & reserved) == reserved {
                assert!(ip!($s).is_reserved());
            } else {
                assert!(!ip!($s).is_reserved());
            }

            if ($mask & shared) == shared {
                assert!(ip!($s).is_shared());
            } else {
                assert!(!ip!($s).is_shared());
            }
        }};
    }

    let unspec: u16 = 1 << 0;
    let loopback: u16 = 1 << 1;
    let private: u16 = 1 << 2;
    let link_local: u16 = 1 << 3;
    let global: u16 = 1 << 4;
    let multicast: u16 = 1 << 5;
    let broadcast: u16 = 1 << 6;
    let documentation: u16 = 1 << 7;
    let benchmarking: u16 = 1 << 8;
    let ietf_protocol_assignment: u16 = 1 << 9;
    let reserved: u16 = 1 << 10;
    let shared: u16 = 1 << 11;

    check!("0.0.0.0", unspec);
    check!("0.0.0.1");
    check!("0.1.0.0");
    check!("10.9.8.7", private);
    check!("127.1.2.3", loopback);
    check!("172.31.254.253", private);
    check!("169.254.253.242", link_local);
    check!("192.0.2.183", documentation);
    check!("192.1.2.183", global);
    check!("192.168.254.253", private);
    check!("198.51.100.0", documentation);
    check!("203.0.113.0", documentation);
    check!("203.2.113.0", global);
    check!("224.0.0.0", global | multicast);
    check!("239.255.255.255", global | multicast);
    check!("255.255.255.255", broadcast);
    check!("198.18.0.0", benchmarking);
    check!("198.18.54.2", benchmarking);
    check!("198.19.255.255", benchmarking);
    check!("192.0.0.0", ietf_protocol_assignment);
    check!("192.0.0.255", ietf_protocol_assignment);
    check!("192.0.0.100", ietf_protocol_assignment);
    check!("240.0.0.0", reserved);
    check!("251.54.1.76", reserved);
    check!("254.255.255.255", reserved);
    check!("100.64.0.0", shared);
    check!("100.127.255.255", shared);
    check!("100.100.100.0", shared);
}

#[test]
fn ipv6_properties() {
    macro_rules! ip {
        ($s:expr) => {
            Ipv6Addr::from_str($s).unwrap()
        };
    }

    macro_rules! check {
        ($s:expr, &[$($octet:expr),*], $mask:expr) => {
            assert_eq!($s, ip!($s).to_string());
            let octets = &[$($octet),*];
            assert_eq!(&ip!($s).octets(), octets);
            assert_eq!(Ipv6Addr::from(*octets), ip!($s));

            let unspecified: u16 = 1 << 0;
            let loopback: u16 = 1 << 1;
            let unique_local: u16 = 1 << 2;
            let global: u16 = 1 << 3;
            let unicast_link_local: u16 = 1 << 4;
            let unicast_global: u16 = 1 << 7;
            let documentation: u16 = 1 << 8;
            let multicast_interface_local: u16 = 1 << 9;
            let multicast_link_local: u16 = 1 << 10;
            let multicast_realm_local: u16 = 1 << 11;
            let multicast_admin_local: u16 = 1 << 12;
            let multicast_site_local: u16 = 1 << 13;
            let multicast_organization_local: u16 = 1 << 14;
            let multicast_global: u16 = 1 << 15;
            let multicast: u16 = multicast_interface_local
                | multicast_admin_local
                | multicast_global
                | multicast_link_local
                | multicast_realm_local
                | multicast_site_local
                | multicast_organization_local;

            if ($mask & unspecified) == unspecified {
                assert!(ip!($s).is_unspecified());
            } else {
                assert!(!ip!($s).is_unspecified());
            }
            if ($mask & loopback) == loopback {
                assert!(ip!($s).is_loopback());
            } else {
                assert!(!ip!($s).is_loopback());
            }
            if ($mask & unique_local) == unique_local {
                assert!(ip!($s).is_unique_local());
            } else {
                assert!(!ip!($s).is_unique_local());
            }
            if ($mask & global) == global {
                assert!(ip!($s).is_global());
            } else {
                assert!(!ip!($s).is_global());
            }
            if ($mask & unicast_link_local) == unicast_link_local {
                assert!(ip!($s).is_unicast_link_local());
            } else {
                assert!(!ip!($s).is_unicast_link_local());
            }
            if ($mask & unicast_global) == unicast_global {
                assert!(ip!($s).is_unicast_global());
            } else {
                assert!(!ip!($s).is_unicast_global());
            }
            if ($mask & documentation) == documentation {
                assert!(ip!($s).is_documentation());
            } else {
                assert!(!ip!($s).is_documentation());
            }
            if ($mask & multicast) != 0 {
                assert!(ip!($s).multicast_scope().is_some());
                assert!(ip!($s).is_multicast());
            } else {
                assert!(ip!($s).multicast_scope().is_none());
                assert!(!ip!($s).is_multicast());
            }
            if ($mask & multicast_interface_local) == multicast_interface_local {
                assert_eq!(ip!($s).multicast_scope().unwrap(),
                           Ipv6MulticastScope::InterfaceLocal);
            }
            if ($mask & multicast_link_local) == multicast_link_local {
                assert_eq!(ip!($s).multicast_scope().unwrap(),
                           Ipv6MulticastScope::LinkLocal);
            }
            if ($mask & multicast_realm_local) == multicast_realm_local {
                assert_eq!(ip!($s).multicast_scope().unwrap(),
                           Ipv6MulticastScope::RealmLocal);
            }
            if ($mask & multicast_admin_local) == multicast_admin_local {
                assert_eq!(ip!($s).multicast_scope().unwrap(),
                           Ipv6MulticastScope::AdminLocal);
            }
            if ($mask & multicast_site_local) == multicast_site_local {
                assert_eq!(ip!($s).multicast_scope().unwrap(),
                           Ipv6MulticastScope::SiteLocal);
            }
            if ($mask & multicast_organization_local) == multicast_organization_local {
                assert_eq!(ip!($s).multicast_scope().unwrap(),
                           Ipv6MulticastScope::OrganizationLocal);
            }
            if ($mask & multicast_global) == multicast_global {
                assert_eq!(ip!($s).multicast_scope().unwrap(),
                           Ipv6MulticastScope::Global);
            }
        }
    }

    let unspecified: u16 = 1 << 0;
    let loopback: u16 = 1 << 1;
    let unique_local: u16 = 1 << 2;
    let global: u16 = 1 << 3;
    let unicast_link_local: u16 = 1 << 4;
    let unicast_global: u16 = 1 << 7;
    let documentation: u16 = 1 << 8;
    let multicast_interface_local: u16 = 1 << 9;
    let multicast_link_local: u16 = 1 << 10;
    let multicast_realm_local: u16 = 1 << 11;
    let multicast_admin_local: u16 = 1 << 12;
    let multicast_site_local: u16 = 1 << 13;
    let multicast_organization_local: u16 = 1 << 14;
    let multicast_global: u16 = 1 << 15;

    check!("::", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], unspecified);

    check!("::1", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], loopback);

    check!("::0.0.0.2", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], global | unicast_global);

    check!("1::", &[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], global | unicast_global);

    check!("fc00::", &[0xfc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], unique_local);

    check!(
        "fdff:ffff::",
        &[0xfd, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        unique_local
    );

    check!(
        "fe80:ffff::",
        &[0xfe, 0x80, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        unicast_link_local
    );

    check!("fe80::", &[0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], unicast_link_local);

    check!(
        "febf:ffff::",
        &[0xfe, 0xbf, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        unicast_link_local
    );

    check!("febf::", &[0xfe, 0xbf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], unicast_link_local);

    check!(
        "febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
        &[
            0xfe, 0xbf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff
        ],
        unicast_link_local
    );

    check!(
        "fe80::ffff:ffff:ffff:ffff",
        &[
            0xfe, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff
        ],
        unicast_link_local
    );

    check!(
        "fe80:0:0:1::",
        &[0xfe, 0x80, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        unicast_link_local
    );

    check!(
        "fec0::",
        &[0xfe, 0xc0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        unicast_global | global
    );

    check!(
        "ff01::",
        &[0xff, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        multicast_interface_local
    );

    check!("ff02::", &[0xff, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], multicast_link_local);

    check!("ff03::", &[0xff, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], multicast_realm_local);

    check!("ff04::", &[0xff, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], multicast_admin_local);

    check!("ff05::", &[0xff, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], multicast_site_local);

    check!(
        "ff08::",
        &[0xff, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        multicast_organization_local
    );

    check!(
        "ff0e::",
        &[0xff, 0xe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        multicast_global | global
    );

    check!(
        "2001:db8:85a3::8a2e:370:7334",
        &[0x20, 1, 0xd, 0xb8, 0x85, 0xa3, 0, 0, 0, 0, 0x8a, 0x2e, 3, 0x70, 0x73, 0x34],
        documentation
    );

    check!(
        "102:304:506:708:90a:b0c:d0e:f10",
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        global | unicast_global
    );
}

#[test]
fn to_socket_addr_socketaddr() {
    let a = sa4(Ipv4Addr::new(77, 88, 21, 11), 12345);
    assert_eq!(Ok(vec![a]), tsa(a));
}

#[test]
fn test_ipv4_to_int() {
    let a = Ipv4Addr::new(0x11, 0x22, 0x33, 0x44);
    assert_eq!(u32::from(a), 0x11223344);
}

#[test]
fn test_int_to_ipv4() {
    let a = Ipv4Addr::new(0x11, 0x22, 0x33, 0x44);
    assert_eq!(Ipv4Addr::from(0x11223344), a);
}

#[test]
fn test_ipv6_to_int() {
    let a = Ipv6Addr::new(0x1122, 0x3344, 0x5566, 0x7788, 0x99aa, 0xbbcc, 0xddee, 0xff11);
    assert_eq!(u128::from(a), 0x112233445566778899aabbccddeeff11u128);
}

#[test]
fn test_int_to_ipv6() {
    let a = Ipv6Addr::new(0x1122, 0x3344, 0x5566, 0x7788, 0x99aa, 0xbbcc, 0xddee, 0xff11);
    assert_eq!(Ipv6Addr::from(0x112233445566778899aabbccddeeff11u128), a);
}

#[test]
fn ipv4_from_constructors() {
    assert_eq!(Ipv4Addr::LOCALHOST, Ipv4Addr::new(127, 0, 0, 1));
    assert!(Ipv4Addr::LOCALHOST.is_loopback());
    assert_eq!(Ipv4Addr::UNSPECIFIED, Ipv4Addr::new(0, 0, 0, 0));
    assert!(Ipv4Addr::UNSPECIFIED.is_unspecified());
    assert_eq!(Ipv4Addr::BROADCAST, Ipv4Addr::new(255, 255, 255, 255));
    assert!(Ipv4Addr::BROADCAST.is_broadcast());
}

#[test]
fn ipv6_from_contructors() {
    assert_eq!(Ipv6Addr::LOCALHOST, Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1));
    assert!(Ipv6Addr::LOCALHOST.is_loopback());
    assert_eq!(Ipv6Addr::UNSPECIFIED, Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0));
    assert!(Ipv6Addr::UNSPECIFIED.is_unspecified());
}

#[test]
fn ipv4_from_octets() {
    assert_eq!(Ipv4Addr::from([127, 0, 0, 1]), Ipv4Addr::new(127, 0, 0, 1))
}

#[test]
fn ipv6_from_segments() {
    let from_u16s =
        Ipv6Addr::from([0x0011, 0x2233, 0x4455, 0x6677, 0x8899, 0xaabb, 0xccdd, 0xeeff]);
    let new = Ipv6Addr::new(0x0011, 0x2233, 0x4455, 0x6677, 0x8899, 0xaabb, 0xccdd, 0xeeff);
    assert_eq!(new, from_u16s);
}

#[test]
fn ipv6_from_octets() {
    let from_u16s =
        Ipv6Addr::from([0x0011, 0x2233, 0x4455, 0x6677, 0x8899, 0xaabb, 0xccdd, 0xeeff]);
    let from_u8s = Ipv6Addr::from([
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ]);
    assert_eq!(from_u16s, from_u8s);
}

#[test]
fn cmp() {
    let v41 = Ipv4Addr::new(100, 64, 3, 3);
    let v42 = Ipv4Addr::new(192, 0, 2, 2);
    let v61 = "2001:db8:f00::1002".parse::<Ipv6Addr>().unwrap();
    let v62 = "2001:db8:f00::2001".parse::<Ipv6Addr>().unwrap();
    assert!(v41 < v42);
    assert!(v61 < v62);

    assert_eq!(v41, IpAddr::V4(v41));
    assert_eq!(v61, IpAddr::V6(v61));
    assert!(v41 != IpAddr::V4(v42));
    assert!(v61 != IpAddr::V6(v62));

    assert!(v41 < IpAddr::V4(v42));
    assert!(v61 < IpAddr::V6(v62));
    assert!(IpAddr::V4(v41) < v42);
    assert!(IpAddr::V6(v61) < v62);

    assert!(v41 < IpAddr::V6(v61));
    assert!(IpAddr::V4(v41) < v61);
}

#[test]
fn is_v4() {
    let ip = IpAddr::V4(Ipv4Addr::new(100, 64, 3, 3));
    assert!(ip.is_ipv4());
    assert!(!ip.is_ipv6());
}

#[test]
fn is_v6() {
    let ip = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x1234, 0x5678));
    assert!(!ip.is_ipv4());
    assert!(ip.is_ipv6());
}

#[test]
fn ipv4_const() {
    // test that the methods of `Ipv4Addr` are usable in a const context

    const IP_ADDRESS: Ipv4Addr = Ipv4Addr::new(127, 0, 0, 1);
    assert_eq!(IP_ADDRESS, Ipv4Addr::LOCALHOST);

    const OCTETS: [u8; 4] = IP_ADDRESS.octets();
    assert_eq!(OCTETS, [127, 0, 0, 1]);

    const IS_UNSPECIFIED: bool = IP_ADDRESS.is_unspecified();
    assert!(!IS_UNSPECIFIED);

    const IS_LOOPBACK: bool = IP_ADDRESS.is_loopback();
    assert!(IS_LOOPBACK);

    const IS_PRIVATE: bool = IP_ADDRESS.is_private();
    assert!(!IS_PRIVATE);

    const IS_LINK_LOCAL: bool = IP_ADDRESS.is_link_local();
    assert!(!IS_LINK_LOCAL);

    const IS_GLOBAL: bool = IP_ADDRESS.is_global();
    assert!(!IS_GLOBAL);

    const IS_SHARED: bool = IP_ADDRESS.is_shared();
    assert!(!IS_SHARED);

    const IS_IETF_PROTOCOL_ASSIGNMENT: bool = IP_ADDRESS.is_ietf_protocol_assignment();
    assert!(!IS_IETF_PROTOCOL_ASSIGNMENT);

    const IS_BENCHMARKING: bool = IP_ADDRESS.is_benchmarking();
    assert!(!IS_BENCHMARKING);

    const IS_RESERVED: bool = IP_ADDRESS.is_reserved();
    assert!(!IS_RESERVED);

    const IS_MULTICAST: bool = IP_ADDRESS.is_multicast();
    assert!(!IS_MULTICAST);

    const IS_BROADCAST: bool = IP_ADDRESS.is_broadcast();
    assert!(!IS_BROADCAST);

    const IS_DOCUMENTATION: bool = IP_ADDRESS.is_documentation();
    assert!(!IS_DOCUMENTATION);

    const IP_V6_COMPATIBLE: Ipv6Addr = IP_ADDRESS.to_ipv6_compatible();
    assert_eq!(
        IP_V6_COMPATIBLE,
        Ipv6Addr::from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 1])
    );

    const IP_V6_MAPPED: Ipv6Addr = IP_ADDRESS.to_ipv6_mapped();
    assert_eq!(
        IP_V6_MAPPED,
        Ipv6Addr::from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 127, 0, 0, 1])
    );
}

#[test]
fn ipv6_const() {
    // test that the methods of `Ipv6Addr` are usable in a const context

    const IP_ADDRESS: Ipv6Addr = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1);
    assert_eq!(IP_ADDRESS, Ipv6Addr::LOCALHOST);

    const SEGMENTS: [u16; 8] = IP_ADDRESS.segments();
    assert_eq!(SEGMENTS, [0, 0, 0, 0, 0, 0, 0, 1]);

    const OCTETS: [u8; 16] = IP_ADDRESS.octets();
    assert_eq!(OCTETS, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);

    const IS_UNSPECIFIED: bool = IP_ADDRESS.is_unspecified();
    assert!(!IS_UNSPECIFIED);

    const IS_LOOPBACK: bool = IP_ADDRESS.is_loopback();
    assert!(IS_LOOPBACK);

    const IS_GLOBAL: bool = IP_ADDRESS.is_global();
    assert!(!IS_GLOBAL);

    const IS_UNIQUE_LOCAL: bool = IP_ADDRESS.is_unique_local();
    assert!(!IS_UNIQUE_LOCAL);

    const IS_UNICAST_LINK_LOCAL: bool = IP_ADDRESS.is_unicast_link_local();
    assert!(!IS_UNICAST_LINK_LOCAL);

    const IS_DOCUMENTATION: bool = IP_ADDRESS.is_documentation();
    assert!(!IS_DOCUMENTATION);

    const IS_UNICAST_GLOBAL: bool = IP_ADDRESS.is_unicast_global();
    assert!(!IS_UNICAST_GLOBAL);

    const MULTICAST_SCOPE: Option<Ipv6MulticastScope> = IP_ADDRESS.multicast_scope();
    assert_eq!(MULTICAST_SCOPE, None);

    const IS_MULTICAST: bool = IP_ADDRESS.is_multicast();
    assert!(!IS_MULTICAST);

    const IP_V4: Option<Ipv4Addr> = IP_ADDRESS.to_ipv4();
    assert_eq!(IP_V4.unwrap(), Ipv4Addr::new(0, 0, 0, 1));
}

#[test]
fn ip_const() {
    // test that the methods of `IpAddr` are usable in a const context

    const IP_ADDRESS: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

    const IS_UNSPECIFIED: bool = IP_ADDRESS.is_unspecified();
    assert!(!IS_UNSPECIFIED);

    const IS_LOOPBACK: bool = IP_ADDRESS.is_loopback();
    assert!(IS_LOOPBACK);

    const IS_GLOBAL: bool = IP_ADDRESS.is_global();
    assert!(!IS_GLOBAL);

    const IS_MULTICAST: bool = IP_ADDRESS.is_multicast();
    assert!(!IS_MULTICAST);

    const IS_IP_V4: bool = IP_ADDRESS.is_ipv4();
    assert!(IS_IP_V4);

    const IS_IP_V6: bool = IP_ADDRESS.is_ipv6();
    assert!(!IS_IP_V6);
}
