// run-pass

#![feature(ip)]
#![feature(const_ipv6)]

use std::net::{Ipv4Addr, Ipv6Addr, Ipv6MulticastScope};

fn main() {
    const IP_ADDRESS : Ipv6Addr = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1);
    assert_eq!(IP_ADDRESS, Ipv6Addr::LOCALHOST);

    const SEGMENTS : [u16; 8] = IP_ADDRESS.segments();
    assert_eq!(SEGMENTS, [0 ,0 ,0 ,0 ,0 ,0 ,0, 1]);

    const OCTETS : [u8; 16] = IP_ADDRESS.octets();
    assert_eq!(OCTETS, [0 ,0 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 1]);

    const IS_UNSPECIFIED : bool = IP_ADDRESS.is_unspecified();
    assert!(!IS_UNSPECIFIED);

    const IS_LOOPBACK : bool = IP_ADDRESS.is_loopback();
    assert!(IS_LOOPBACK);

    const IS_GLOBAL : bool = IP_ADDRESS.is_global();
    assert!(!IS_GLOBAL);

    const IS_UNIQUE_LOCAL : bool = IP_ADDRESS.is_unique_local();
    assert!(!IS_UNIQUE_LOCAL);

    const IS_UNICAST_LINK_LOCAL_STRICT : bool = IP_ADDRESS.is_unicast_link_local_strict();
    assert!(!IS_UNICAST_LINK_LOCAL_STRICT);

    const IS_UNICAST_LINK_LOCAL : bool = IP_ADDRESS.is_unicast_link_local();
    assert!(!IS_UNICAST_LINK_LOCAL);

    const IS_UNICAST_SITE_LOCAL : bool = IP_ADDRESS.is_unicast_site_local();
    assert!(!IS_UNICAST_SITE_LOCAL);

    const IS_DOCUMENTATION : bool = IP_ADDRESS.is_documentation();
    assert!(!IS_DOCUMENTATION);

    const IS_UNICAST_GLOBAL : bool = IP_ADDRESS.is_unicast_global();
    assert!(!IS_UNICAST_GLOBAL);

    const MULTICAST_SCOPE : Option<Ipv6MulticastScope> = IP_ADDRESS.multicast_scope();
    assert_eq!(MULTICAST_SCOPE, None);

    const IS_MULTICAST : bool = IP_ADDRESS.is_multicast();
    assert!(!IS_MULTICAST);

    const IP_V4 : Option<Ipv4Addr> = IP_ADDRESS.to_ipv4();
    assert_eq!(IP_V4.unwrap(), Ipv4Addr::new(0, 0, 0, 1));
}
