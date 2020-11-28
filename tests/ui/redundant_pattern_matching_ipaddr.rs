// run-rustfix

#![warn(clippy::all)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(unused_must_use, clippy::needless_bool, clippy::match_like_matches_macro)]

use std::net::{
    IpAddr::{self, V4, V6},
    Ipv4Addr, Ipv6Addr,
};

fn main() {
    let ipaddr: IpAddr = V4(Ipv4Addr::LOCALHOST);
    if let V4(_) = &ipaddr {}

    if let V4(_) = V4(Ipv4Addr::LOCALHOST) {}

    if let V6(_) = V6(Ipv6Addr::LOCALHOST) {}

    while let V4(_) = V4(Ipv4Addr::LOCALHOST) {}

    while let V6(_) = V6(Ipv6Addr::LOCALHOST) {}

    if V4(Ipv4Addr::LOCALHOST).is_ipv4() {}

    if V6(Ipv6Addr::LOCALHOST).is_ipv6() {}

    if let V4(ipaddr) = V4(Ipv4Addr::LOCALHOST) {
        println!("{}", ipaddr);
    }

    match V4(Ipv4Addr::LOCALHOST) {
        V4(_) => true,
        V6(_) => false,
    };

    match V4(Ipv4Addr::LOCALHOST) {
        V4(_) => false,
        V6(_) => true,
    };

    match V6(Ipv6Addr::LOCALHOST) {
        V4(_) => false,
        V6(_) => true,
    };

    match V6(Ipv6Addr::LOCALHOST) {
        V4(_) => true,
        V6(_) => false,
    };

    let _ = if let V4(_) = V4(Ipv4Addr::LOCALHOST) {
        true
    } else {
        false
    };

    ipaddr_const();

    let _ = if let V4(_) = gen_ipaddr() {
        1
    } else if let V6(_) = gen_ipaddr() {
        2
    } else {
        3
    };
}

fn gen_ipaddr() -> IpAddr {
    V4(Ipv4Addr::LOCALHOST)
}

const fn ipaddr_const() {
    if let V4(_) = V4(Ipv4Addr::LOCALHOST) {}

    if let V6(_) = V6(Ipv6Addr::LOCALHOST) {}

    while let V4(_) = V4(Ipv4Addr::LOCALHOST) {}

    while let V6(_) = V6(Ipv6Addr::LOCALHOST) {}

    match V4(Ipv4Addr::LOCALHOST) {
        V4(_) => true,
        V6(_) => false,
    };

    match V6(Ipv6Addr::LOCALHOST) {
        V4(_) => false,
        V6(_) => true,
    };
}
