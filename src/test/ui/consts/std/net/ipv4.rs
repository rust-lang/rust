// run-pass

#![feature(ip)]
#![feature(const_ipv4)]

use std::net::Ipv4Addr;

fn main() {
    const IP_ADDRESS: Ipv4Addr = Ipv4Addr::new(127, 0, 0, 1);
    assert_eq!(IP_ADDRESS, Ipv4Addr::LOCALHOST);

    const OCTETS: [u8; 4] = IP_ADDRESS.octets();
    assert_eq!(OCTETS, [127, 0, 0, 1]);

    const IS_UNSPECIFIED : bool = IP_ADDRESS.is_unspecified();
    assert!(!IS_UNSPECIFIED);

    const IS_LOOPBACK : bool = IP_ADDRESS.is_loopback();
    assert!(IS_LOOPBACK);

    const IS_PRIVATE : bool = IP_ADDRESS.is_private();
    assert!(!IS_PRIVATE);

    const IS_LINK_LOCAL : bool = IP_ADDRESS.is_link_local();
    assert!(!IS_LINK_LOCAL);

    const IS_SHARED : bool = IP_ADDRESS.is_shared();
    assert!(!IS_SHARED);

    const IS_IETF_PROTOCOL_ASSIGNMENT : bool = IP_ADDRESS.is_ietf_protocol_assignment();
    assert!(!IS_IETF_PROTOCOL_ASSIGNMENT);

    const IS_BENCHMARKING : bool = IP_ADDRESS.is_benchmarking();
    assert!(!IS_BENCHMARKING);

    const IS_MULTICAST : bool = IP_ADDRESS.is_multicast();
    assert!(!IS_MULTICAST);

    const IS_DOCUMENTATION : bool = IP_ADDRESS.is_documentation();
    assert!(!IS_DOCUMENTATION);
}
