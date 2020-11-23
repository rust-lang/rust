// run-pass

use std::net::{IpAddr, Ipv4Addr};

fn main() {
    const IP_ADDRESS : IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

    const IS_IP_V4 : bool = IP_ADDRESS.is_ipv4();
    assert!(IS_IP_V4);

    const IS_IP_V6 : bool = IP_ADDRESS.is_ipv6();
    assert!(!IS_IP_V6);
}
